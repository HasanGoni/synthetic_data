"""
Epic 2 US 2.5 — paste synthetic defect patch onto a real OK background.

Uses ``cv2.seamlessClone`` (Poisson) or simple alpha blending inside the mask.
"""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np

BlendMode = Literal["poisson", "alpha"]


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """2D uint8/uint16 grayscale or BGR → uint8 BGR."""
    if img.ndim == 3 and img.shape[2] >= 3:
        x = img[:, :, :3]
        if x.dtype == np.uint8:
            return x
        if x.dtype == np.uint16:
            return (x.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(x)
    if img.ndim != 2:
        raise ValueError("expected H×W or H×W×3 image")
    if img.dtype == np.uint16:
        g = (img.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    else:
        g = img.astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _bgr_u8_to_match(background_like: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 result back to same layout/dtype as ``background_like``."""
    if background_like.ndim == 2:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if background_like.shape[2] == 1:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    if background_like.dtype == np.uint16:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 * 65535.0
        return g.astype(np.uint16)
    return bgr


def paste_defect_on_background(
    background: np.ndarray,
    defect_patch: np.ndarray,
    defect_mask: np.ndarray,
    center_xy: Tuple[int, int],
    mode: BlendMode = "poisson",
    alpha: float = 0.9,
) -> np.ndarray:
    """
    Paste ``defect_patch`` onto ``background`` with center at ``center_xy`` (x, y).

    ``defect_mask`` is uint8, 255 = defect pixels from the patch. Patch and mask same H×W.
    """
    if defect_patch.shape[:2] != defect_mask.shape[:2]:
        raise ValueError("defect_patch and defect_mask must match in H×W")
    m = (defect_mask > 127).astype(np.uint8) * 255
    if int(m.sum()) == 0:
        return background.copy()

    h, w = defect_patch.shape[:2]
    cx, cy = center_xy
    x0 = int(cx - w // 2)
    y0 = int(cy - h // 2)
    x1, y1 = x0 + w, y0 + h
    if x0 < 0 or y0 < 0 or x1 > background.shape[1] or y1 > background.shape[0]:
        raise ValueError(
            f"patch does not fit: need roi [{x0}:{x1}, {y0}:{y1}] inside "
            f"{background.shape[1]}×{background.shape[0]}"
        )

    if mode == "alpha":
        return _alpha_blend(background, defect_patch, m, x0, y0, x1, y1, alpha)

    dst_full = _to_bgr_u8(background)
    src = _to_bgr_u8(defect_patch)
    center_pt = (int(cx), int(cy))

    try:
        blended_bgr = cv2.seamlessClone(src, dst_full, m, center_pt, cv2.NORMAL_CLONE)
    except cv2.error:
        return _alpha_blend(background, defect_patch, m, x0, y0, x1, y1, alpha=1.0)

    return _bgr_u8_to_match(background, blended_bgr)


def _alpha_blend(
    background: np.ndarray,
    defect_patch: np.ndarray,
    mask_255: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha: float,
) -> np.ndarray:
    mf = (mask_255 > 127).astype(np.float32)
    if mf.ndim == 2:
        mf = mf[..., None]
    out = background.astype(np.float32)
    roi = out[y0:y1, x0:x1]
    p = defect_patch.astype(np.float32)
    if p.ndim == 2:
        p = p[..., None]
    if roi.ndim == 2:
        roi = roi[..., None]
    if roi.shape[2] != p.shape[2]:
        if p.shape[2] == 1 and roi.shape[2] >= 1:
            p = np.repeat(p, roi.shape[2], axis=2)
        elif roi.shape[2] == 1 and p.shape[2] > 1:
            roi = np.repeat(roi, p.shape[2], axis=2)
    a = (alpha * mf).clip(0, 1)
    merged = roi * (1 - a) + p * a
    merged = np.squeeze(merged) if background.ndim == 2 else merged
    if background.dtype == np.uint16:
        merged = np.clip(merged, 0, 65535).astype(np.uint16)
    else:
        merged = np.clip(merged, 0, 255).astype(background.dtype)
    result = background.copy()
    if result.ndim == 2:
        result[y0:y1, x0:x1] = merged.squeeze() if merged.ndim > 2 else merged
    else:
        result[y0:y1, x0:x1] = merged
    return result
