"""
Epic 4 -- Failure categorisation and reporting.

Analyses model predictions at the image level, categorises each sample
into TP / FP / FN / TN, and generates a structured failure report with
visual examples of the worst predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Minimum fraction of positive pixels required to consider the image-level
# prediction or ground truth as "positive" (void present).
_POSITIVE_AREA_THRESHOLD = 1e-4


# -- Helpers ------------------------------------------------------------------


def _collate_with_paths(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that stacks images/masks and preserves paths."""
    images = torch.stack([s["image"] for s in batch])
    masks = [s["mask"] for s in batch]
    has_masks = all(m is not None for m in masks)
    stacked_masks = torch.stack(masks) if has_masks else None
    paths = [s["path"] for s in batch]
    return {"image": images, "mask": stacked_masks, "path": paths}


def _image_level_category(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> str:
    """Classify a single sample at the image level.

    An image is considered positive if the fraction of foreground pixels
    exceeds :data:`_POSITIVE_AREA_THRESHOLD`.

    Returns one of ``'TP'``, ``'FP'``, ``'FN'``, ``'TN'``.
    """
    pred_pos = pred_mask.mean() > _POSITIVE_AREA_THRESHOLD
    true_pos = true_mask.mean() > _POSITIVE_AREA_THRESHOLD

    if pred_pos and true_pos:
        return "TP"
    if pred_pos and not true_pos:
        return "FP"
    if not pred_pos and true_pos:
        return "FN"
    return "TN"


def _pixel_iou(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """Pixel-level IoU between two binary masks."""
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    if union == 0.0:
        return 1.0
    return float(intersection / union)


# -- Public API ---------------------------------------------------------------


@torch.no_grad()
def categorize_failures(
    model: nn.Module,
    dataset: Any,
    device: str = "cuda",
    threshold: float = 0.5,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run inference and categorise every sample at the image level.

    Each sample is labelled TP, FP, FN, or TN based on whether the model
    and ground truth agree on void presence.  For FP and FN cases
    additional diagnostics (predicted area, true area, error region) are
    computed.

    Args:
        model:      Segmentation model returning logits ``[B, 1, H, W]``.
        dataset:    Dataset with ``image``, ``mask``, and ``path`` keys.
        device:     Torch device string.
        threshold:  Binarisation threshold (after sigmoid).
        batch_size: Mini-batch size for inference.

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``image_path``, ``category``, ``pred_area``, ``true_area``,
        ``iou``, ``notes``.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_with_paths,
    )

    rows: list[dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"]
        paths = batch["path"]

        if masks is None:
            logger.warning("Skipping batch with no masks in failure analysis.")
            continue
        masks = masks.to(device)

        logits = model(images)
        # Handle models that return (seg_logits, domain_logits) tuples.
        if isinstance(logits, tuple):
            logits = logits[0]
        # Match spatial resolution to mask.
        if logits.shape[2:] != masks.shape[2:]:
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[2:], mode="bilinear", align_corners=False,
            )

        probs = torch.sigmoid(logits)
        pred_bin = (probs >= threshold).float()

        # Process each sample individually.
        for i in range(images.shape[0]):
            pred_np = pred_bin[i, 0].cpu().numpy()
            mask_np = masks[i, 0].cpu().numpy()

            category = _image_level_category(pred_np, mask_np)
            pred_area = float(pred_np.mean())
            true_area = float(mask_np.mean())
            iou = _pixel_iou(pred_np, mask_np)

            # Build human-readable notes for failure cases.
            notes = ""
            if category == "FP":
                notes = (
                    f"False positive: model predicted {pred_area:.4f} void fraction "
                    f"but true is {true_area:.4f}."
                )
            elif category == "FN":
                notes = (
                    f"False negative: model missed void region "
                    f"(true area={true_area:.4f}, pred area={pred_area:.4f})."
                )

            rows.append(
                {
                    "image_path": paths[i],
                    "category": category,
                    "pred_area": pred_area,
                    "true_area": true_area,
                    "iou": iou,
                    "notes": notes,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["image_path", "category", "pred_area", "true_area", "iou", "notes"],
    )
    logger.info(
        "Failure categorisation: TP=%d  FP=%d  FN=%d  TN=%d",
        (df["category"] == "TP").sum(),
        (df["category"] == "FP").sum(),
        (df["category"] == "FN").sum(),
        (df["category"] == "TN").sum(),
    )
    return df


def generate_failure_report(
    failures_df: pd.DataFrame,
    output_dir: str,
    model: Optional[nn.Module] = None,
    dataset: Optional[Any] = None,
    device: str = "cuda",
    n_visual: int = 20,
) -> None:
    """Generate a structured failure report to *output_dir*.

    Outputs:
        1. ``failures.csv`` -- full categorised DataFrame.
        2. ``summary.txt`` -- aggregate statistics.
        3. ``visuals/`` -- PNG overlays for the *n_visual* worst failures
           (ranked by ascending IoU among TP/FP/FN samples).

    If *model* and *dataset* are provided, the function re-runs inference
    to produce visual overlays.  Otherwise only the CSV and summary are
    written.

    Args:
        failures_df: Output of :func:`categorize_failures`.
        output_dir:  Directory where the report artefacts are saved.
        model:       Optional model for generating visual overlays.
        dataset:     Optional dataset aligned with *failures_df* rows.
        device:      Torch device string.
        n_visual:    Number of worst-case visual examples to save.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- 1. Save full CSV -------------------------------------------------
    csv_path = out / "failures.csv"
    failures_df.to_csv(csv_path, index=False)
    logger.info("Failures CSV saved to %s", csv_path)

    # -- 2. Summary statistics --------------------------------------------
    total = len(failures_df)
    counts = failures_df["category"].value_counts().to_dict()
    mean_iou = failures_df["iou"].mean() if total > 0 else 0.0

    # Metrics restricted to positive samples (TP + FP + FN).
    positives = failures_df[failures_df["category"].isin(["TP", "FP", "FN"])]
    mean_iou_pos = positives["iou"].mean() if len(positives) > 0 else 0.0

    # Compute image-level precision and recall.
    n_tp = counts.get("TP", 0)
    n_fp = counts.get("FP", 0)
    n_fn = counts.get("FN", 0)
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)

    summary_lines = [
        "Failure Analysis Summary",
        "=" * 40,
        f"Total samples analysed: {total}",
        "",
        "Category counts:",
        f"  TP (true positive):  {n_tp}",
        f"  FP (false positive): {n_fp}",
        f"  FN (false negative): {n_fn}",
        f"  TN (true negative):  {counts.get('TN', 0)}",
        "",
        f"Image-level precision: {precision:.4f}",
        f"Image-level recall:    {recall:.4f}",
        "",
        f"Mean IoU (all samples):      {mean_iou:.4f}",
        f"Mean IoU (positive samples): {mean_iou_pos:.4f}",
        "",
    ]

    # Per-category area statistics.
    for cat in ["TP", "FP", "FN"]:
        cat_df = failures_df[failures_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        summary_lines.append(f"{cat} statistics (n={len(cat_df)}):")
        summary_lines.append(f"  Mean pred area:  {cat_df['pred_area'].mean():.4f}")
        summary_lines.append(f"  Mean true area:  {cat_df['true_area'].mean():.4f}")
        summary_lines.append(f"  Mean IoU:        {cat_df['iou'].mean():.4f}")
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    summary_path = out / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    # -- 3. Visual examples of worst failures -----------------------------
    if model is None or dataset is None:
        logger.info("Skipping visual generation (model or dataset not provided).")
        return

    visuals_dir = out / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Select worst failures: lowest IoU among non-TN samples.
    non_tn = failures_df[failures_df["category"] != "TN"].copy()
    if len(non_tn) == 0:
        logger.info("No non-TN samples to visualise.")
        return

    worst = non_tn.nsmallest(min(n_visual, len(non_tn)), "iou")

    # Build a path -> dataset-index lookup so we can reload images.
    path_to_idx: dict[str, int] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        path_to_idx[sample["path"]] = idx

    model.eval()
    saved_count = 0

    for _, row in worst.iterrows():
        img_path = row["image_path"]
        ds_idx = path_to_idx.get(img_path)
        if ds_idx is None:
            logger.warning("Could not find dataset index for %s", img_path)
            continue

        sample = dataset[ds_idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"]

        with torch.no_grad():
            logits = model(image_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            if mask is not None and logits.shape[2:] != mask.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=mask.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Prepare display arrays.
        # Use the first channel of the normalised image as a grayscale preview.
        img_display = sample["image"][0].numpy()
        img_display = (
            (img_display - img_display.min())
            / (img_display.max() - img_display.min() + 1e-8)
        )

        mask_display = mask[0].numpy() if mask is not None else np.zeros_like(pred_prob)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_display, cmap="gray")
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title(f"Prediction ({row['category']}, IoU={row['iou']:.3f})")
        axes[2].axis("off")

        fig.suptitle(Path(img_path).name, fontsize=10)
        fig.tight_layout()

        save_name = f"failure_{saved_count:03d}_{row['category']}_{Path(img_path).stem}.png"
        fig.savefig(str(visuals_dir / save_name), dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_count += 1

    logger.info("Saved %d visual failure examples to %s", saved_count, visuals_dir)
