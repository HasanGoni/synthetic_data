"""
Epic 4 — DANN adversarial training loop.

Trains a DANNModel (SharedEncoder + UNetDecoder + DomainClassifier with GRL)
using mixed source/target batches.  Source samples contribute both
segmentation and domain loss; target samples contribute domain loss only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from udm_epic4.data.domain_sampler import DomainBatchSampler
from udm_epic4.data.multi_domain_dataset import build_datasets_from_config
from udm_epic4.evaluation.metrics import compute_f1, compute_iou
from udm_epic4.models.dann import DANNModel
from udm_epic4.training.lambda_scheduler import dann_lambda_schedule
from udm_epic4.training.train_baseline import bce_dice_loss

console = Console()

# ------------------------------------------------------------------
# Validation (source-domain only — we have masks for source)
# ------------------------------------------------------------------


def _validate_dann(
    model: DANNModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate segmentation quality on the source validation set."""
    model.eval()

    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # lambda=0 during eval — domain head is irrelevant
            seg_logits, _ = model(images, lambda_val=0.0)
            loss = bce_dice_loss(seg_logits, masks)
            running_loss += loss.item() * images.size(0)

            preds_bin = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            all_preds.append(preds_bin)
            all_targets.append(masks_np)

    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_targets_arr = np.concatenate(all_targets, axis=0)

    val_loss = running_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
    val_f1 = compute_f1(all_preds_arr, all_targets_arr)
    val_iou = compute_iou(all_preds_arr, all_targets_arr)

    return {"val_loss": val_loss, "val_f1": val_f1, "val_iou": val_iou}


# ------------------------------------------------------------------
# Training entry-point
# ------------------------------------------------------------------


def train_dann_from_yaml(config_path: str | Path) -> Path:
    """
    Train a DANN model from a YAML config.

    The config must contain at least::

        training:
          epochs, lr, weight_decay, grad_clip_max_norm,
          seg_loss_weight, domain_loss_weight,
          lambda_max, checkpoint_every, batch_size, num_workers,
          output_dir
        model:
          backbone_name, pretrained, in_chans

    Each training batch is a 50/50 mix of source (with masks) and target
    (no masks) samples assembled by :class:`DomainBatchSampler`.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Path to the best checkpoint (highest source-validation F1).
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg["training"]
    model_cfg = cfg.get("model", {})

    epochs: int = train_cfg["epochs"]
    lr: float = train_cfg["lr"]
    weight_decay: float = train_cfg.get("weight_decay", 1e-4)
    grad_clip: float = train_cfg.get("grad_clip_max_norm", 1.0)
    seg_weight: float = train_cfg.get("seg_loss_weight", 1.0)
    domain_weight: float = train_cfg.get("domain_loss_weight", 1.0)
    lambda_max: float = train_cfg.get("lambda_max", 1.0)
    ckpt_every: int = train_cfg.get("checkpoint_every", 5)
    batch_size: int = train_cfg.get("batch_size", 8)
    num_workers: int = train_cfg.get("num_workers", 4)
    output_dir = Path(train_cfg.get("output_dir", "outputs/dann"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/] {device}")

    # ---- Datasets -------------------------------------------------------
    datasets = build_datasets_from_config(cfg)
    combined_train = datasets["combined_train"]
    source_val = datasets["source_val"]

    domain_sampler = DomainBatchSampler(
        dataset=combined_train,
        batch_size=batch_size,
        drop_last=True,
    )
    train_loader = DataLoader(
        combined_train,
        batch_sampler=domain_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        source_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Model ----------------------------------------------------------
    model = DANNModel(
        backbone_name=model_cfg.get("backbone_name", "convnext_tiny"),
        pretrained=model_cfg.get("pretrained", True),
        in_chans=model_cfg.get("in_chans", 3),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    domain_bce = nn.BCEWithLogitsLoss()

    # ---- Training loop --------------------------------------------------
    best_f1 = -1.0
    best_ckpt_path = output_dir / "best_dann.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        progress = (epoch - 1) / max(epochs - 1, 1)
        lambda_val = dann_lambda_schedule(progress, lambda_max=lambda_max)

        running_seg_loss = 0.0
        running_dom_loss = 0.0
        running_dom_correct = 0
        running_dom_total = 0
        n_source_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            domain_labels_raw = batch["domain_label"].to(device)  # 0=source, 1=target
            source_mask = domain_labels_raw == 0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                seg_logits, domain_logits = model(images, lambda_val=lambda_val)

                # Segmentation loss — source samples only
                if source_mask.any():
                    source_masks = batch["mask"].to(device)
                    seg_logits_src = seg_logits[source_mask]
                    masks_src = source_masks[source_mask]
                    loss_seg = bce_dice_loss(seg_logits_src, masks_src)
                else:
                    loss_seg = torch.tensor(0.0, device=device)

                # Domain loss — all samples
                domain_targets = domain_labels_raw.float().unsqueeze(1)
                loss_domain = domain_bce(domain_logits, domain_targets)

                loss_total = seg_weight * loss_seg + domain_weight * loss_domain

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Bookkeeping
            bs = images.size(0)
            n_src = int(source_mask.sum().item())
            running_seg_loss += loss_seg.item() * max(n_src, 1)
            running_dom_loss += loss_domain.item() * bs
            n_source_samples += n_src

            dom_preds = (torch.sigmoid(domain_logits) > 0.5).long().squeeze(1)
            running_dom_correct += (dom_preds == domain_labels_raw).sum().item()
            running_dom_total += bs

            pbar.set_postfix(
                seg=f"{loss_seg.item():.3f}",
                dom=f"{loss_domain.item():.3f}",
                lam=f"{lambda_val:.3f}",
            )

        avg_seg = running_seg_loss / max(n_source_samples, 1)
        avg_dom = running_dom_loss / max(running_dom_total, 1)
        dom_acc = running_dom_correct / max(running_dom_total, 1)

        # ---- Validation (source only) -----------------------------------
        val_metrics = _validate_dann(model, val_loader, device)

        # ---- Logging (rich table) ---------------------------------------
        table = Table(title=f"Epoch {epoch}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("seg_loss", f"{avg_seg:.4f}")
        table.add_row("domain_loss", f"{avg_dom:.4f}")
        table.add_row("lambda", f"{lambda_val:.4f}")
        table.add_row("domain_acc", f"{dom_acc:.4f}")
        table.add_row("source_val_f1", f"{val_metrics['val_f1']:.4f}")
        table.add_row("source_val_iou", f"{val_metrics['val_iou']:.4f}")
        console.print(table)

        # ---- Checkpointing ----------------------------------------------
        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_f1,
                    "lambda": lambda_val,
                },
                best_ckpt_path,
            )
            console.print(f"[bold yellow]  -> New best model saved (F1={best_f1:.4f})[/]")

        if epoch % ckpt_every == 0:
            periodic_path = output_dir / f"dann_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["val_f1"],
                    "lambda": lambda_val,
                },
                periodic_path,
            )

    console.print(f"[bold green]Training complete. Best checkpoint:[/] {best_ckpt_path}")
    return best_ckpt_path
