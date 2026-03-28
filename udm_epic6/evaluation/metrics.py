"""
Epic 6 — Evaluation metrics for bond wire defect detection.

Reuses the pixel-level ``compute_f1`` and ``compute_iou`` from Epic 4 and
adds wire-specific metrics:

* :func:`wire_detection_rate` — fraction of true wires overlapping a prediction.
* :func:`defect_classification_accuracy` — accuracy of defect type labels.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor


# ── Re-exported pixel-level metrics (same logic as Epic 4) ───────────────────


def _binarise(pred: Tensor, threshold: float) -> Tensor:
    if pred.min() < 0.0 or pred.max() > 1.0:
        pred = torch.sigmoid(pred)
    return (pred >= threshold).float()


def _confusion_counts(pred_bin: Tensor, target: Tensor):
    target_bin = target.float()
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1.0 - target_bin)).sum()
    fn = ((1.0 - pred_bin) * target_bin).sum()
    tn = ((1.0 - pred_bin) * (1.0 - target_bin)).sum()
    return tp, fp, fn, tn


def compute_f1(pred: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Binary F1 score over all pixels (see Epic 4 ``compute_f1``)."""
    pred_bin = _binarise(pred, threshold)
    tp, fp, fn, _ = _confusion_counts(pred_bin, target)
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float((2.0 * tp / denom).item())


def compute_iou(pred: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-Union over all pixels (see Epic 4 ``compute_iou``)."""
    pred_bin = _binarise(pred, threshold)
    tp, fp, fn, _ = _confusion_counts(pred_bin, target)
    denom = tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float((tp / denom).item())


# ── Wire-specific metrics ────────────────────────────────────────────────────


def wire_detection_rate(
    pred_masks: List[np.ndarray],
    true_masks: List[np.ndarray],
    wire_masks: List[np.ndarray],
    overlap_threshold: float = 0.3,
) -> float:
    """Fraction of individual wires that are detected by the predictions.

    A wire is considered *detected* if the IoU between the predicted mask
    restricted to the wire region and the ground-truth mask restricted to
    the wire region exceeds *overlap_threshold*.

    Args:
        pred_masks:        List of predicted binary masks (H, W), one per sample.
        true_masks:        List of ground-truth binary masks (H, W).
        wire_masks:        List of per-wire binary masks (H, W) — one mask per
                           individual wire across all samples.
        overlap_threshold: Minimum IoU to count a wire as detected.

    Returns:
        Detection rate in [0, 1].
    """
    if len(wire_masks) == 0:
        return 1.0

    detected = 0
    total = 0

    for pred, true, wire in zip(pred_masks, true_masks, wire_masks):
        wire_region = wire.astype(bool)
        if not wire_region.any():
            continue

        total += 1
        pred_in_wire = pred.astype(bool) & wire_region
        true_in_wire = true.astype(bool) & wire_region

        intersection = (pred_in_wire & true_in_wire).sum()
        union = (pred_in_wire | true_in_wire).sum()

        if union == 0:
            # Both empty in this wire region — counts as detected
            detected += 1
        elif intersection / union >= overlap_threshold:
            detected += 1

    return detected / max(total, 1)


def defect_classification_accuracy(
    pred_types: Sequence[str],
    true_types: Sequence[str],
) -> float:
    """Compute accuracy of defect type classification.

    Args:
        pred_types: Predicted defect type labels (e.g. ``"bend"``, ``"none"``).
        true_types: Ground-truth defect type labels.

    Returns:
        Accuracy in [0, 1].
    """
    if len(pred_types) == 0:
        return 1.0

    correct = sum(p == t for p, t in zip(pred_types, true_types))
    return correct / len(pred_types)
