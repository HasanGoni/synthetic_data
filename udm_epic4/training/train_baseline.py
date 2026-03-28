"""
Epic 4 — Source-only baseline segmentation trainer.

Trains a SharedEncoder + UNetDecoder on the *source* domain only (no domain
adaptation).  Serves as the lower-bound comparison for DANN.
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
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from udm_epic4.data.multi_domain_dataset import build_datasets_from_config
from udm_epic4.evaluation.metrics import compute_f1, compute_iou
from udm_epic4.models.decoder import UNetDecoder
from udm_epic4.models.encoder import SharedEncoder

console = Console()

# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------


def bce_dice_loss(pred_logits: Tensor, target: Tensor) -> Tensor:
    """
    Combined binary cross-entropy + Dice loss.

    Args:
        pred_logits: Raw logits of shape ``[B, 1, H, W]``.
        target:      Binary ground-truth mask ``[B, 1, H, W]`` (0 or 1).

    Returns:
        Scalar loss (mean BCE + (1 - Dice)).
    """
    bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, target)

    probs = torch.sigmoid(pred_logits)
    smooth = 1e-6
    intersection = (probs * target).sum(dim=(2, 3))
    cardinality = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    dice_loss = 1.0 - dice.mean()

    return bce + dice_loss


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def _validate(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation pass, returning loss / F1 / IoU."""
    encoder.eval()
    decoder.eval()

    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            features = encoder(images)
            logits = decoder(features)
            loss = bce_dice_loss(logits, masks)
            running_loss += loss.item() * images.size(0)

            preds_bin = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
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


def train_baseline_from_yaml(config_path: str | Path) -> Path:
    """
    Train a source-only segmentation model from a YAML config.

    Config keys consumed (under ``training``)::

        epochs, lr, weight_decay, grad_clip_max_norm,
        checkpoint_every, batch_size, num_workers,
        output_dir

    Config keys consumed (under ``model``)::

        backbone_name, pretrained, in_chans

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Path to the best checkpoint (highest validation F1).
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
    ckpt_every: int = train_cfg.get("checkpoint_every", 5)
    batch_size: int = train_cfg.get("batch_size", 8)
    num_workers: int = train_cfg.get("num_workers", 4)
    output_dir = Path(train_cfg.get("output_dir", "outputs/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/] {device}")

    # ---- Datasets -------------------------------------------------------
    datasets = build_datasets_from_config(cfg)
    source_train = datasets["source_train"]
    source_val = datasets["source_val"]

    train_loader = DataLoader(
        source_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        source_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Model ----------------------------------------------------------
    encoder = SharedEncoder(
        backbone_name=model_cfg.get("backbone_name", "convnext_tiny"),
        pretrained=model_cfg.get("pretrained", True),
        in_chans=model_cfg.get("in_chans", 3),
    ).to(device)

    decoder = UNetDecoder(
        encoder_channels=encoder.feature_channels,
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # ---- Training loop --------------------------------------------------
    best_f1 = -1.0
    best_ckpt_path = output_dir / "best_baseline.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                features = encoder(images)
                logits = decoder(features)
                loss = bce_dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader.dataset), 1)  # type: ignore[arg-type]

        # ---- Validation -------------------------------------------------
        val_metrics = _validate(encoder, decoder, val_loader, device)

        # ---- Logging (rich table) ---------------------------------------
        table = Table(title=f"Epoch {epoch}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("train_loss", f"{train_loss:.4f}")
        table.add_row("val_loss", f"{val_metrics['val_loss']:.4f}")
        table.add_row("val_f1", f"{val_metrics['val_f1']:.4f}")
        table.add_row("val_iou", f"{val_metrics['val_iou']:.4f}")
        console.print(table)

        # ---- Checkpointing ----------------------------------------------
        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_f1,
                },
                best_ckpt_path,
            )
            console.print(f"[bold yellow]  -> New best model saved (F1={best_f1:.4f})[/]")

        if epoch % ckpt_every == 0:
            periodic_path = output_dir / f"baseline_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["val_f1"],
                },
                periodic_path,
            )

    console.print(f"[bold green]Training complete. Best checkpoint:[/] {best_ckpt_path}")
    return best_ckpt_path
