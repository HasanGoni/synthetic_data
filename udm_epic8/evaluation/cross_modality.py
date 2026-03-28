"""
Epic 8 -- Cross-modality evaluation and real-vs-synthetic comparison.

Functions
---------
cross_modality_report     Build a unified metrics DataFrame from per-modality results.
compare_real_vs_synthetic Compare real and synthetic image quality for a modality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Cross-modality report ─────────────────────────────────────────────────────


def cross_modality_report(results_per_modality: Dict[str, dict]) -> pd.DataFrame:
    """Build a unified metrics table from per-modality evaluation results.

    Parameters
    ----------
    results_per_modality : dict
        Mapping from modality name to a dict of metrics.  Each dict should
        contain at least some of the standard keys: ``f1``, ``iou``, ``dice``,
        ``n_samples``, ``fid``, ``ssim``, ``psnr``.  Missing keys are filled
        with ``NaN``.

    Returns
    -------
    pd.DataFrame
        One row per modality with columns for every metric found across all
        modalities, sorted by modality name.

    Examples
    --------
    >>> results = {
    ...     "xray": {"f1": 0.91, "iou": 0.84, "n_samples": 500},
    ...     "aoi":  {"f1": 0.87, "iou": 0.79, "n_samples": 300},
    ... }
    >>> df = cross_modality_report(results)
    >>> list(df.columns)
    ['modality', 'f1', 'iou', 'n_samples']
    """
    if not results_per_modality:
        return pd.DataFrame(columns=["modality"])

    # Collect all metric keys across modalities
    all_keys: set[str] = set()
    for metrics in results_per_modality.values():
        all_keys.update(metrics.keys())
    all_keys_sorted = sorted(all_keys)

    rows: List[Dict[str, Any]] = []
    for modality in sorted(results_per_modality.keys()):
        metrics = results_per_modality[modality]
        row: Dict[str, Any] = {"modality": modality}
        for key in all_keys_sorted:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)

    columns = ["modality"] + all_keys_sorted
    df = pd.DataFrame(rows, columns=columns)
    return df


# ── Real vs synthetic comparison ──────────────────────────────────────────────


def compare_real_vs_synthetic(
    real_dir: str,
    synthetic_dir: str,
    modality: str,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """Compare real and synthetic image distributions for a given modality.

    Computes lightweight image quality metrics (mean intensity, standard
    deviation, SSIM where available) between images in *real_dir* and
    *synthetic_dir*.

    Parameters
    ----------
    real_dir : str
        Directory containing real images (PNG).
    synthetic_dir : str
        Directory containing synthetic images (PNG).
    modality : str
        Modality label included in the result dict.
    max_samples : int
        Maximum number of images to sample from each directory.

    Returns
    -------
    dict
        Metrics dict with keys ``modality``, ``real_mean``, ``real_std``,
        ``synth_mean``, ``synth_std``, ``mean_diff``, ``std_diff``, and
        optionally ``mean_ssim``.
    """
    from PIL import Image

    real_path = Path(real_dir)
    synth_path = Path(synthetic_dir)

    real_images = sorted(real_path.glob("*.png"))[:max_samples]
    synth_images = sorted(synth_path.glob("*.png"))[:max_samples]

    if not real_images:
        raise FileNotFoundError(f"No PNG images found in real directory: {real_dir}")
    if not synth_images:
        raise FileNotFoundError(f"No PNG images found in synthetic directory: {synthetic_dir}")

    # Compute intensity statistics
    real_stats = _compute_intensity_stats(real_images)
    synth_stats = _compute_intensity_stats(synth_images)

    result: Dict[str, Any] = {
        "modality": modality,
        "n_real": len(real_images),
        "n_synthetic": len(synth_images),
        "real_mean": real_stats["mean"],
        "real_std": real_stats["std"],
        "synth_mean": synth_stats["mean"],
        "synth_std": synth_stats["std"],
        "mean_diff": abs(real_stats["mean"] - synth_stats["mean"]),
        "std_diff": abs(real_stats["std"] - synth_stats["std"]),
    }

    # Try to compute SSIM if scikit-image is available and image counts match
    try:
        from skimage.metrics import structural_similarity as ssim

        n_compare = min(len(real_images), len(synth_images), 50)
        ssim_scores: list[float] = []
        for i in range(n_compare):
            r = np.array(Image.open(real_images[i]).convert("L")).astype(np.float64)
            s = np.array(Image.open(synth_images[i]).convert("L")).astype(np.float64)
            # Resize synthetic to match real if shapes differ
            if r.shape != s.shape:
                s = np.array(
                    Image.open(synth_images[i]).convert("L").resize(
                        (r.shape[1], r.shape[0])
                    )
                ).astype(np.float64)
            score = ssim(r, s, data_range=255.0)
            ssim_scores.append(score)
        result["mean_ssim"] = float(np.mean(ssim_scores))
    except ImportError:
        logger.debug("scikit-image not available; skipping SSIM computation")

    return result


def _compute_intensity_stats(image_paths: list[Path]) -> Dict[str, float]:
    """Compute mean and std of pixel intensities across a list of images."""
    from PIL import Image

    values: list[float] = []
    for p in image_paths:
        arr = np.array(Image.open(p).convert("L")).astype(np.float64)
        values.append(arr.mean())

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
