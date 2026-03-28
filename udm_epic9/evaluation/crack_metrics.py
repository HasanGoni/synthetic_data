"""
Epic 9 — Evaluation metrics for crack detection and segmentation.

Provides crack-specific metrics:

- **Detection rate**: fraction of ground-truth cracks that are detected
- **Crack length error**: relative error in skeleton length
- **Full evaluation**: runs a model on a dataset and returns a DataFrame
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ── Skeleton length utility ────────────────────────────────────────────────


def _skeleton_length(mask: np.ndarray) -> float:
    """Compute total crack length as skeleton pixel count.

    Uses morphological thinning to extract the skeleton, then counts
    non-zero pixels as a proxy for crack length.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8 or bool), shape ``(H, W)``.

    Returns
    -------
    float
        Skeleton length in pixels.
    """
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0
    skel = skeletonize(binary > 0)
    return float(skel.sum())


# ── Metrics ────────────────────────────────────────────────────────────────


def crack_detection_rate(
    pred_masks: Union[List[np.ndarray], np.ndarray],
    true_masks: Union[List[np.ndarray], np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Fraction of ground-truth cracks that are detected.

    A ground-truth crack is considered "detected" if the predicted mask
    has any overlap with the true crack region (IoU > 0 after thresholding).

    Parameters
    ----------
    pred_masks : array-like
        Predicted masks (float probabilities or binary), each ``(H, W)``.
    true_masks : array-like
        Ground-truth binary masks, each ``(H, W)``.
    threshold : float
        Binarisation threshold for predicted masks.

    Returns
    -------
    float
        Detection rate in [0, 1].
    """
    if isinstance(pred_masks, np.ndarray) and pred_masks.ndim == 2:
        pred_masks = [pred_masks]
        true_masks = [true_masks]

    n_total = 0
    n_detected = 0

    for pred, true in zip(pred_masks, true_masks):
        pred_bin = (pred >= threshold).astype(np.uint8)
        true_bin = (true > 0).astype(np.uint8)

        # Only count samples that actually contain cracks
        if true_bin.sum() == 0:
            continue

        n_total += 1
        intersection = (pred_bin & true_bin).sum()
        if intersection > 0:
            n_detected += 1

    if n_total == 0:
        return 1.0  # No cracks to detect — vacuously true

    return n_detected / n_total


def crack_length_error(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    """Relative error in total crack length (skeleton length).

    Computes the skeletons of both masks and returns the relative
    difference in pixel count.

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask ``(H, W)``.
    true_mask : np.ndarray
        Ground-truth binary mask ``(H, W)``.

    Returns
    -------
    float
        Relative error ``|pred_len - true_len| / max(true_len, 1)``.
    """
    pred_len = _skeleton_length(pred_mask)
    true_len = _skeleton_length(true_mask)
    denom = max(true_len, 1.0)
    return abs(pred_len - true_len) / denom


def evaluate_crack_dataset(
    model: torch.nn.Module,
    dataset: "torch.utils.data.Dataset",
    device: str = "cuda",
    threshold: float = 0.5,
    batch_size: int = 4,
) -> pd.DataFrame:
    """Evaluate a segmentation model on a crack dataset.

    Runs inference on every sample and computes per-sample detection and
    length error metrics, returned as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model that takes image tensor and returns logits/probs.
    dataset : Dataset
        Must return dicts with ``'image'`` and ``'mask'`` keys.
    device : str
        Torch device string.
    threshold : float
        Binarisation threshold for predictions.
    batch_size : int
        Inference batch size.

    Returns
    -------
    pd.DataFrame
        Columns: ``index``, ``crack_type``, ``has_crack``, ``detected``,
        ``length_error``, ``iou``.
    """
    from torch.utils.data import DataLoader

    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks_true = batch["mask"]
            crack_types = batch["crack_type"]

            logits = model(images)
            if isinstance(logits, dict):
                logits = logits.get("out", logits.get("logits", list(logits.values())[0]))
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(images.shape[0]):
                pred = probs[i, 0]
                true = masks_true[i, 0].numpy()
                pred_bin = (pred >= threshold).astype(np.uint8)
                true_bin = (true > 0.5).astype(np.uint8)

                has_crack = true_bin.sum() > 0
                detected = bool((pred_bin & true_bin).sum() > 0) if has_crack else True

                # IoU
                intersection = (pred_bin & true_bin).sum()
                union = (pred_bin | true_bin).sum()
                iou = float(intersection / max(union, 1))

                length_err = crack_length_error(pred_bin, true_bin) if has_crack else 0.0

                records.append({
                    "index": sample_idx,
                    "crack_type": crack_types[i] if isinstance(crack_types, list) else str(crack_types[i]),
                    "has_crack": has_crack,
                    "detected": detected,
                    "length_error": length_err,
                    "iou": iou,
                })
                sample_idx += 1

    return pd.DataFrame(records)
