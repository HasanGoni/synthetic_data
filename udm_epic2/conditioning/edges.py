"""
Epic 2 US 2.2 — edge / boundary maps for conditioning (ControlNet-style).

Uses Canny on the binary void mask; optional dilation thickens strokes for training stability.
"""

from __future__ import annotations

import numpy as np
import cv2


def edge_map_from_mask(mask_uint8: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Convert a void mask (255 = defect) to a single-channel edge map (uint8).

    """
    if mask_uint8.ndim != 2:
        raise ValueError("mask must be 2D")
    m = (mask_uint8 > 127).astype(np.uint8) * 255
    if int(m.sum()) == 0:
        return np.zeros_like(m)

    edges = cv2.Canny(m, 30, 100)
    if thickness > 1:
        k = max(3, thickness * 2 + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, ker)
    return edges
