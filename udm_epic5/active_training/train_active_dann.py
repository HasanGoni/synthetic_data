"""
Epic 5 — Active DANN training loop.

Fine-tunes a DANN model (from Epic 4) using:
  - Source domain samples (all labeled)
  - Target domain samples selected by the active learning oracle (labeled)
  - Remaining target domain samples (unlabeled, domain loss only)

Key difference from Epic 4: segmentation loss is computed on *both*
source and labeled-target samples, giving the model direct supervision
from the target distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from udm_epic4.data.multi_domain_dataset import DomainDataset, build_datasets_from_config
from udm_epic4.evaluation.metrics import compute_f1, compute_iou
from udm_epic4.models.dann import DANNModel
from udm_epic4.training.lambda_scheduler import dann_lambda_schedule
from udm_epic4.training.train_baseline import bce_dice_loss

logger = logging.getLogger(__name__)
console = Console()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_labeled_target_dataset(
    images_dir: str | Path,
    masks_dir: str | Path,
    image_size: tuple[int, int] = (512, 512),
    domain_label: int = 1,
) -> DomainDataset:
    """Build a :class:`DomainDataset` for the labeled target subset.

    These are the target images selected by the active-learning oracle
    for which ground-truth masks have been provided.

    Args:
        images_dir:   Directory containing the selected target images.
        masks_dir:    Directory containing corresponding human-annotated masks.
        image_size:   Target (height, width) for resizing.
        domain_label: Integer domain identifier for the target domain.

    Returns:
        A :class:`DomainDataset` with both images and masks.
    """
    return DomainDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        domain_label=domain_label,
        image_size=image_size,
    )


def _validate(
    model: DANNModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate segmentation quality on a labeled set.

    Runs inference with ``lambda_val=0`` (domain head inactive) and returns
    aggregate loss, F1, and IoU.

    Args:
        model:  The DANN model to evaluate.
        loader: DataLoader yielding dicts with ``image`` and ``mask`` keys.
        device: Torch device.

    Returns:
        Dictionary with ``val_loss``, ``val_f1``, ``val_iou``.
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            if masks is None:
                continue
            masks = masks.to(device)

            seg_logits, _ = model(images, lambda_val=0.0)
            loss = bce_dice_loss(seg_logits, masks)
            running_loss += loss.item() * images.size(0)

            preds_bin = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            all_preds.append(preds_bin)
            all_targets.append(masks_np)

    if len(all_preds) == 0:
        return {"val_loss": float("inf"), "val_f1": 0.0, "val_iou": 0.0}

    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_targets_arr = np.concatenate(all_targets, axis=0)

    n_samples = max(all_preds_arr.shape[0], 1)
    val_loss = running_loss / n_samples
    val_f1 = compute_f1(all_preds_arr, all_targets_arr)
    val_iou = compute_iou(all_preds_arr, all_targets_arr)

    return {"val_loss": val_loss, "val_f1": val_f1, "val_iou": val_iou}


def _collate_with_flags(batch: list[dict]) -> dict[str, Any]:
    """Custom collate that tracks which samples have masks.

    Returns a dict with an additional ``has_mask`` boolean tensor of shape
    ``[B]``, which is True for samples that have a non-None mask.
    Samples without masks get a zero-filled mask placeholder so the batch
    can be stacked uniformly.
    """
    images = torch.stack([s["image"] for s in batch])

    has_mask_flags: list[bool] = []
    masks: list[torch.Tensor] = []
    for s in batch:
        if s["mask"] is not None:
            masks.append(s["mask"])
            has_mask_flags.append(True)
        else:
            # Placeholder zeros matching expected shape [1, H, W]
            h, w = s["image"].shape[1], s["image"].shape[2]
            masks.append(torch.zeros(1, h, w, dtype=torch.float32))
            has_mask_flags.append(False)

    masks_stacked = torch.stack(masks)
    has_mask = torch.tensor(has_mask_flags, dtype=torch.bool)

    # Preserve domain labels (0=source, 1=target)
    domain_labels = torch.tensor(
        [s.get("domain", s.get("domain_label", 0)) for s in batch],
        dtype=torch.long,
    )

    return {
        "image": images,
        "mask": masks_stacked,
        "has_mask": has_mask,
        "domain_label": domain_labels,
    }


# ------------------------------------------------------------------
# Main training entry-point
# ------------------------------------------------------------------


def train_active_dann_from_yaml(config_path: str | Path) -> Path:
    """
    Train a DANN model with active-learning-selected labeled target samples.

    The YAML config extends the Epic 4 schema with an ``active_learning``
    section::

        active_learning:
          labeled_target_images: /path/to/selected/images
          labeled_target_masks:  /path/to/selected/masks
          base_checkpoint: null   # path to Epic 4 pre-trained DANN (optional)

    The segmentation loss is computed over *all* samples that have masks
    (source train + labeled target), while the domain-adversarial loss
    covers the entire batch (source + labeled target + unlabeled target).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Path to the best checkpoint (highest target-validation F1).
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg["training"]
    model_cfg = cfg.get("model", {})
    active_cfg = cfg.get("active_learning", {})

    # ---- Hyper-parameters -----------------------------------------------
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
    output_dir = Path(train_cfg.get("output_dir", "outputs/active_dann"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/] {device}")

    # ---- Datasets -------------------------------------------------------
    datasets = build_datasets_from_config(cfg.get("data", cfg))
    source_train = datasets["source_train"]
    source_val = datasets["source_val"]
    target_unlabeled_list = datasets.get("targets", [])

    # Labeled target subset (from active learning selection)
    labeled_target_images = active_cfg.get("labeled_target_images")
    labeled_target_masks = active_cfg.get("labeled_target_masks")

    labeled_target_ds: Optional[DomainDataset] = None
    if labeled_target_images and labeled_target_masks:
        labeled_target_ds = _build_labeled_target_dataset(
            images_dir=labeled_target_images,
            masks_dir=labeled_target_masks,
            image_size=image_size,
            domain_label=active_cfg.get("target_domain_label", 1),
        )
        console.print(
            f"[bold cyan]Labeled target samples:[/] {len(labeled_target_ds)}"
        )

    # Target validation set (for tracking target-domain F1)
    eval_datasets = datasets.get("evaluation", [])
    target_val_ds = eval_datasets[0] if eval_datasets else None

    # Combined training dataset: source + labeled_target + unlabeled_target
    # Each sample carries domain_label and has_mask info via the collate fn.
    train_parts = [source_train]
    if labeled_target_ds is not None and len(labeled_target_ds) > 0:
        train_parts.append(labeled_target_ds)
    for tgt_ds in target_unlabeled_list:
        train_parts.append(tgt_ds)

    combined_train = ConcatDataset(train_parts)

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_with_flags,
        drop_last=True,
    )

    # Validation loaders
    source_val_loader = DataLoader(
        source_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_with_flags,
    )
    target_val_loader: Optional[DataLoader] = None
    if target_val_ds is not None:
        target_val_loader = DataLoader(
            target_val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_with_flags,
        )

    # ---- Model ----------------------------------------------------------
    model = DANNModel(
        backbone=model_cfg.get("backbone_name", "convnext_tiny"),
        pretrained=model_cfg.get("pretrained", True),
        in_chans=model_cfg.get("in_chans", 3),
    ).to(device)

    base_checkpoint = active_cfg.get("base_checkpoint")
    if base_checkpoint is not None:
        ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        console.print(
            f"[bold yellow]Loaded base checkpoint:[/] {base_checkpoint} "
            f"(epoch {ckpt.get('epoch', '?')}, F1={ckpt.get('val_f1', '?')})"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    domain_bce = nn.BCEWithLogitsLoss()

    # ---- Training loop --------------------------------------------------
    best_target_f1 = -1.0
    best_ckpt_path = output_dir / "best_active_dann.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        progress = (epoch - 1) / max(epochs - 1, 1)
        lambda_val = dann_lambda_schedule(progress, lambda_max=lambda_max)

        running_seg_loss = 0.0
        running_dom_loss = 0.0
        n_labeled_samples = 0
        n_total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)
            domain_labels = batch["domain_label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                seg_logits, domain_logits = model(images, lambda_val=lambda_val)

                # Segmentation loss — all samples WITH masks
                # (source + labeled target)
                if has_mask.any():
                    seg_logits_labeled = seg_logits[has_mask]
                    masks_labeled = masks[has_mask]
                    loss_seg = bce_dice_loss(seg_logits_labeled, masks_labeled)
                else:
                    loss_seg = torch.tensor(0.0, device=device)

                # Domain loss — all samples (source=0, target=1)
                domain_targets = domain_labels.float().unsqueeze(1)
                loss_domain = domain_bce(domain_logits, domain_targets)

                loss_total = (
                    seg_weight * loss_seg
                    + domain_weight * lambda_val * loss_domain
                )

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Bookkeeping
            bs = images.size(0)
            n_labeled = int(has_mask.sum().item())
            running_seg_loss += loss_seg.item() * max(n_labeled, 1)
            running_dom_loss += loss_domain.item() * bs
            n_labeled_samples += n_labeled
            n_total_samples += bs

            pbar.set_postfix(
                seg=f"{loss_seg.item():.3f}",
                dom=f"{loss_domain.item():.3f}",
                lam=f"{lambda_val:.3f}",
            )

        avg_seg = running_seg_loss / max(n_labeled_samples, 1)
        avg_dom = running_dom_loss / max(n_total_samples, 1)

        # ---- Validation (source + target if available) ------------------
        source_metrics = _validate(model, source_val_loader, device)

        target_metrics: Dict[str, float] = {
            "val_loss": float("inf"),
            "val_f1": 0.0,
            "val_iou": 0.0,
        }
        if target_val_loader is not None:
            target_metrics = _validate(model, target_val_loader, device)

        # ---- Logging (rich table) ---------------------------------------
        table = Table(title=f"Epoch {epoch}/{epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("seg_loss", f"{avg_seg:.4f}")
        table.add_row("domain_loss", f"{avg_dom:.4f}")
        table.add_row("lambda", f"{lambda_val:.4f}")
        table.add_row("source_f1", f"{source_metrics['val_f1']:.4f}")
        table.add_row("target_f1", f"{target_metrics['val_f1']:.4f}")
        table.add_row("source_iou", f"{source_metrics['val_iou']:.4f}")
        table.add_row("target_iou", f"{target_metrics['val_iou']:.4f}")
        console.print(table)

        # ---- Checkpointing (best by target F1) --------------------------
        current_target_f1 = target_metrics["val_f1"]
        if current_target_f1 > best_target_f1:
            best_target_f1 = current_target_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "source_f1": source_metrics["val_f1"],
                    "target_f1": best_target_f1,
                    "lambda": lambda_val,
                },
                best_ckpt_path,
            )
            console.print(
                f"[bold yellow]  -> New best model saved "
                f"(target F1={best_target_f1:.4f})[/]"
            )

        if epoch % ckpt_every == 0:
            periodic_path = output_dir / f"active_dann_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "source_f1": source_metrics["val_f1"],
                    "target_f1": current_target_f1,
                    "lambda": lambda_val,
                },
                periodic_path,
            )

    console.print(
        f"[bold green]Active DANN training complete. "
        f"Best checkpoint:[/] {best_ckpt_path}"
    )
    return best_ckpt_path


# ------------------------------------------------------------------
# Simpler fine-tune alternative (no adversarial loss)
# ------------------------------------------------------------------


def finetune_with_labels(
    base_checkpoint: str | Path,
    labeled_target_dir: str,
    labeled_target_masks: str,
    config_path: str | Path,
) -> Path:
    """
    Fine-tune a DANN on source + labeled target without adversarial loss.

    This is a simpler alternative to :func:`train_active_dann_from_yaml`
    for quick experiments: the domain classifier is disabled, and only the
    segmentation objective is optimised on the union of source and labeled
    target data.

    Args:
        base_checkpoint:      Path to the pre-trained DANN checkpoint.
        labeled_target_dir:   Directory with selected target images.
        labeled_target_masks: Directory with corresponding target masks.
        config_path:          YAML config (used for data paths, image_size,
                              training hyper-parameters).

    Returns:
        Path to the best checkpoint (highest target validation F1).
    """
    config_path = Path(config_path)
    base_checkpoint = Path(base_checkpoint)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg["training"]
    model_cfg = cfg.get("model", {})

    epochs: int = train_cfg.get("finetune_epochs", train_cfg.get("epochs", 20))
    lr: float = train_cfg.get("finetune_lr", train_cfg.get("lr", 1e-4))
    weight_decay: float = train_cfg.get("weight_decay", 1e-4)
    grad_clip: float = train_cfg.get("grad_clip_max_norm", 1.0)
    batch_size: int = train_cfg.get("batch_size", 8)
    num_workers: int = train_cfg.get("num_workers", 4)
    output_dir = Path(train_cfg.get("output_dir", "outputs/finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/] {device}")

    # ---- Datasets -------------------------------------------------------
    datasets = build_datasets_from_config(cfg.get("data", cfg))
    source_train = datasets["source_train"]
    source_val = datasets["source_val"]
    eval_datasets = datasets.get("evaluation", [])
    target_val_ds = eval_datasets[0] if eval_datasets else None

    labeled_target_ds = _build_labeled_target_dataset(
        images_dir=labeled_target_dir,
        masks_dir=labeled_target_masks,
        image_size=image_size,
    )
    console.print(
        f"[bold cyan]Fine-tuning with {len(labeled_target_ds)} "
        f"labeled target samples[/]"
    )

    combined = ConcatDataset([source_train, labeled_target_ds])
    train_loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_with_flags,
        drop_last=True,
    )

    target_val_loader: Optional[DataLoader] = None
    if target_val_ds is not None:
        target_val_loader = DataLoader(
            target_val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_with_flags,
        )

    # ---- Model ----------------------------------------------------------
    model = DANNModel(
        backbone=model_cfg.get("backbone_name", "convnext_tiny"),
        pretrained=model_cfg.get("pretrained", True),
        in_chans=model_cfg.get("in_chans", 3),
    ).to(device)

    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    console.print(
        f"[bold yellow]Loaded base checkpoint:[/] {base_checkpoint}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # ---- Training loop (segmentation only) ------------------------------
    best_target_f1 = -1.0
    best_ckpt_path = output_dir / "best_finetune.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            if not has_mask.any():
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type, enabled=(device.type == "cuda")
            ):
                seg_logits, _ = model(images, lambda_val=0.0)
                seg_logits_labeled = seg_logits[has_mask]
                masks_labeled = masks[has_mask]
                loss = bce_dice_loss(seg_logits_labeled, masks_labeled)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            n_labeled = int(has_mask.sum().item())
            running_loss += loss.item() * n_labeled
            n_samples += n_labeled
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = running_loss / max(n_samples, 1)

        # ---- Validation -------------------------------------------------
        target_f1 = 0.0
        if target_val_loader is not None:
            target_metrics = _validate(model, target_val_loader, device)
            target_f1 = target_metrics["val_f1"]

        console.print(
            f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
            f"target_f1={target_f1:.4f}"
        )

        if target_f1 > best_target_f1:
            best_target_f1 = target_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "target_f1": best_target_f1,
                },
                best_ckpt_path,
            )
            console.print(
                f"[bold yellow]  -> New best finetune model "
                f"(target F1={best_target_f1:.4f})[/]"
            )

    console.print(
        f"[bold green]Fine-tuning complete. Best checkpoint:[/] {best_ckpt_path}"
    )
    return best_ckpt_path
