"""
Statistics from binary void masks (real or synthetic).

Mask convention: 8-bit grayscale, void pixels > 127.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def void_metrics_from_mask(mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute void metrics from a 2D binary mask (255 = void, 0 = background).

    Examples
    --------
    >>> m = np.zeros((64, 64), dtype=np.uint8)
    >>> m[20:40, 20:40] = 255
    >>> out = void_metrics_from_mask(m)
    >>> out["n_voids"]
    1
    >>> out["void_area_fraction"] > 0
    True
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    h, w = mask.shape
    n_pixels = h * w
    binary = (mask > 127).astype(np.uint8)
    void_pixels = int(binary.sum())
    area_fraction = void_pixels / n_pixels if n_pixels else 0.0
    num_labels, _ = cv2.connectedComponents(binary)
    n_voids = max(0, num_labels - 1)
    return {
        "height": h,
        "width": w,
        "n_voids": n_voids,
        "void_pixels": void_pixels,
        "void_area_fraction": float(area_fraction),
    }


def analyze_mask_path(path: Path) -> Dict[str, Any]:
    """Load a mask image from disk and return metrics plus file name."""
    p = Path(path)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {p}")
    out = void_metrics_from_mask(m)
    out["file"] = p.name
    out["path"] = str(p.resolve())
    return out


def summarize_mask_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate mean/std/min/max for n_voids and void_area_fraction."""
    if not records:
        return {"n_masks": 0}

    def _stat(key: str) -> Dict[str, float]:
        vals = [float(r[key]) for r in records if key in r]
        if not vals:
            return {}
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "n_masks": len(records),
        "n_voids": _stat("n_voids"),
        "void_area_fraction": _stat("void_area_fraction"),
    }


def scan_mask_directory(
    mask_dir: Path,
    glob: str = "*.png",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scan all masks under ``mask_dir`` matching ``glob``.

    Returns (per_image records, summary dict).
    """
    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob(glob))
    records: List[Dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            records.append(analyze_mask_path(p))
        except (FileNotFoundError, ValueError):
            continue
    summary = summarize_mask_records(records)
    summary["mask_dir"] = str(mask_dir.resolve())
    summary["glob"] = glob
    return records, summary


def write_stats_json(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write analysis results to JSON."""
    payload: Dict[str, Any] = {
        "summary": summary,
        "per_image": records,
    }
    if extra:
        payload["meta"] = extra
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
