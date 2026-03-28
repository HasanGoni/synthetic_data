"""
Epic 4 — Segmentation metrics per domain.

Provides pixel-level binary segmentation metrics (F1, IoU, Dice) and
convenience functions to evaluate a trained model across one or more
domain-specific datasets, returning results as a :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Pixel-level binary metrics ──────────────────────────────────────────────


def _binarise(pred: Tensor, threshold: float) -> Tensor:
    """Apply sigmoid (if needed) and threshold to produce binary mask.

    If *pred* contains values outside [0, 1] it is treated as raw logits
    and passed through ``torch.sigmoid`` first.
    """
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    return (pred >= threshold).float()


def _confusion_counts(pred_bin: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (TP, FP, FN, TN) summed over all spatial dimensions."""
    target_bin = target.float()
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1.0 - target_bin)).sum()
    fn = ((1.0 - pred_bin) * target_bin).sum()
    tn = ((1.0 - pred_bin) * (1.0 - target_bin)).sum()
    return tp, fp, fn, tn


def compute_f1(pred: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Compute binary F1 score over all pixels.

    Args:
        pred:      Model output logits or probabilities, any shape broadcastable
                   with *target*.
        target:    Binary ground-truth mask (0/1), same shape as *pred*.
        threshold: Decision boundary applied after optional sigmoid.

    Returns:
        F1 score as a Python float in [0, 1].
    """
    pred_bin = _binarise(pred, threshold)
    tp, fp, fn, _tn = _confusion_counts(pred_bin, target)
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 1.0  # both pred and target are entirely negative
    return float((2.0 * tp / denom).item())


def compute_iou(pred: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection-over-Union (Jaccard index) over all pixels.

    Args:
        pred:      Model output logits or probabilities.
        target:    Binary ground-truth mask (0/1).
        threshold: Decision boundary applied after optional sigmoid.

    Returns:
        IoU as a Python float in [0, 1].
    """
    pred_bin = _binarise(pred, threshold)
    tp, fp, fn, _tn = _confusion_counts(pred_bin, target)
    denom = tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float((tp / denom).item())


def compute_dice(pred: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Compute the Dice coefficient (F1 at the pixel level).

    Numerically identical to :func:`compute_f1` for binary masks, but kept
    as a separate entry point for clarity in metric naming conventions.

    .. math::
        \\text{Dice} = \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}

    Args:
        pred:      Model output logits or probabilities.
        target:    Binary ground-truth mask (0/1).
        threshold: Decision boundary applied after optional sigmoid.

    Returns:
        Dice coefficient as a Python float in [0, 1].
    """
    pred_bin = _binarise(pred, threshold)
    tp, fp, fn, _tn = _confusion_counts(pred_bin, target)
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float((2.0 * tp / denom).item())


# ── Model-level evaluation ──────────────────────────────────────────────────


def _collate_skip_none_masks(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that stacks images and masks, skipping None masks."""
    images = torch.stack([s["image"] for s in batch])
    masks = [s["mask"] for s in batch]
    has_masks = all(m is not None for m in masks)
    stacked_masks = torch.stack(masks) if has_masks else None
    domains = torch.tensor([s["domain"] for s in batch], dtype=torch.long)
    paths = [s["path"] for s in batch]
    return {"image": images, "mask": stacked_masks, "domain": domains, "path": paths}


@torch.no_grad()
def evaluate_model_on_domain(
    model: torch.nn.Module,
    dataset: Any,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict:
    """Run inference on *dataset* and compute aggregate segmentation metrics.

    The model is expected to accept an image tensor ``[B, C, H, W]`` and
    return logits of shape ``[B, 1, H, W]``.  If the logit spatial size
    differs from the mask, bilinear interpolation is used to match.

    Args:
        model:      Segmentation model (encoder + decoder pipeline).
        dataset:    A :class:`~torch.utils.data.Dataset` returning dicts with
                    keys ``image``, ``mask``, ``domain``, ``path``
                    (see :class:`DomainDataset`).
        device:     Torch device string.
        batch_size: Evaluation mini-batch size.

    Returns:
        Dictionary with keys ``f1``, ``iou``, ``dice``, ``n_samples``, and
        ``domain_name``.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_skip_none_masks,
    )

    all_preds: list[Tensor] = []
    all_masks: list[Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        if masks is None:
            logger.warning("Skipping batch with no masks during evaluation.")
            continue
        masks = masks.to(device)

        logits = model(images)
        # Ensure spatial dimensions match the ground-truth mask.
        if logits.shape[2:] != masks.shape[2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False,
            )

        all_preds.append(logits.cpu())
        all_masks.append(masks.cpu())

    if len(all_preds) == 0:
        logger.error("No valid batches found for evaluation.")
        return {
            "f1": 0.0,
            "iou": 0.0,
            "dice": 0.0,
            "n_samples": 0,
            "domain_name": _resolve_domain_name(dataset),
        }

    preds_cat = torch.cat(all_preds, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)

    return {
        "f1": compute_f1(preds_cat, masks_cat),
        "iou": compute_iou(preds_cat, masks_cat),
        "dice": compute_dice(preds_cat, masks_cat),
        "n_samples": int(preds_cat.shape[0]),
        "domain_name": _resolve_domain_name(dataset),
    }


def _resolve_domain_name(dataset: Any) -> str:
    """Best-effort extraction of a human-readable domain name."""
    # DomainDataset stores images_dir which usually encodes the site name.
    if hasattr(dataset, "images_dir"):
        return str(dataset.images_dir.stem)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "images_dir"):
        # torch Subset wraps the original dataset
        return str(dataset.dataset.images_dir.stem)
    if hasattr(dataset, "domain_label"):
        return f"domain_{dataset.domain_label}"
    return "unknown"


def evaluate_all_domains(
    model: torch.nn.Module,
    eval_datasets: list,
    device: str = "cuda",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Evaluate *model* on every dataset in *eval_datasets*.

    Args:
        model:         Segmentation model.
        eval_datasets: List of datasets, each following the
                       :class:`DomainDataset` interface.
        device:        Torch device string.
        batch_size:    Evaluation mini-batch size.

    Returns:
        :class:`pandas.DataFrame` with one row per domain and columns
        ``domain_name``, ``f1``, ``iou``, ``dice``, ``n_samples``.
    """
    rows: list[dict] = []
    for ds in eval_datasets:
        result = evaluate_model_on_domain(model, ds, device=device, batch_size=batch_size)
        rows.append(result)
        logger.info(
            "Domain %-20s  F1=%.4f  IoU=%.4f  Dice=%.4f  (n=%d)",
            result["domain_name"],
            result["f1"],
            result["iou"],
            result["dice"],
            result["n_samples"],
        )

    df = pd.DataFrame(rows, columns=["domain_name", "f1", "iou", "dice", "n_samples"])
    return df
