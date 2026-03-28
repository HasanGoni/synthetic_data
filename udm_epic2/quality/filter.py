"""Epic 2 US 2.4 — cheap quality gates before saving generated samples (blur / blank rejection)."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def laplacian_variance(gray_u8: np.ndarray) -> float:
    """Variance of Laplacian — higher usually means sharper image."""
    if gray_u8.ndim != 2:
        raise ValueError("expected single-channel uint8 image")
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def passes_quality_gate(
    image_bgr_or_gray: np.ndarray,
    min_laplacian_var: float = 5.0,
    max_laplacian_var: float = 1e7,
) -> Tuple[bool, float]:
    """
    Return (ok, score). Reject near-blank or extreme failure modes using Laplacian variance only.
    """
    if image_bgr_or_gray.ndim == 3:
        g = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        g = image_bgr_or_gray
    if g.dtype != np.uint8:
        g = cv2.convertScaleAbs(g)
    v = laplacian_variance(g)
    ok = min_laplacian_var <= v <= max_laplacian_var
    return ok, v
