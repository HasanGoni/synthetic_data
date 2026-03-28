"""
Epic 9 — Render cracks onto USM (Ultrasonic Microscopy) style images.

In USM imaging, cracks appear as bright lines due to acoustic impedance
mismatch at the crack surface.  Deep cracks can show edge diffraction
effects (bright edges, darker center).  This module provides functions
to render physically-plausible crack appearances on synthetic USM
backgrounds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from udm_epic9.models.crack_types import (
    delamination_crack,
    die_crack,
    mold_crack,
    substrate_crack,
)

logger = logging.getLogger(__name__)

# Mapping from name to generator function
_CRACK_GENERATORS = {
    "die_crack": die_crack,
    "substrate_crack": substrate_crack,
    "mold_crack": mold_crack,
    "delamination_crack": delamination_crack,
}


# ── USM background synthesis ──────────────────────────────────────────────


def _generate_usm_background(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.03,
) -> np.ndarray:
    """Generate a synthetic USM background image.

    The background simulates the appearance of a bonded semiconductor layer
    in ultrasonic microscopy — a relatively uniform gray-level field with
    low-frequency spatial variation and mild speckle noise.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    rng : numpy.random.Generator, optional
        Random generator.
    noise_std : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    np.ndarray
        Float32 image in [0, 1] range, shape ``(height, width)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base intensity: uniform with slight low-freq variation
    base_level = rng.uniform(0.3, 0.6)
    bg = np.full((height, width), base_level, dtype=np.float32)

    # Low-frequency spatial variation (simulates package geometry)
    low_freq = rng.normal(0, 0.08, size=(height // 16 + 1, width // 16 + 1)).astype(np.float32)
    low_freq = cv2.resize(low_freq, (width, height), interpolation=cv2.INTER_CUBIC)
    bg += low_freq

    # Add mild speckle noise (USM characteristic)
    speckle = rng.normal(0, noise_std, size=(height, width)).astype(np.float32)
    bg += speckle

    return np.clip(bg, 0.0, 1.0)


# ── Crack rendering on USM ────────────────────────────────────────────────


def render_crack_on_usm(
    background: np.ndarray,
    crack_mask: np.ndarray,
    crack_intensity: float = 0.7,
    edge_effect: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Overlay a crack onto a USM background image.

    Cracks in USM imaging appear as bright lines due to the acoustic
    impedance mismatch at the air-filled crack.  This function modulates
    the crack intensity along the mask and optionally adds edge diffraction
    effects.

    Parameters
    ----------
    background : np.ndarray
        USM background image, float32 in [0, 1], shape ``(H, W)``.
    crack_mask : np.ndarray
        Binary crack mask, uint8 0/255 or float 0/1, shape ``(H, W)``.
    crack_intensity : float
        Peak intensity of the crack relative to background (0..1).
    edge_effect : bool
        If ``True``, add edge-bright / center-dark diffraction effect
        for wider crack regions (simulating deep cracks).
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    np.ndarray
        Float32 image in [0, 1] with crack overlaid.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = background.copy().astype(np.float32)
    h, w = result.shape[:2]

    # Normalise mask to float [0, 1]
    mask_f = crack_mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f = mask_f / 255.0

    # Intensity modulation along crack (slight randomness per pixel)
    intensity_map = mask_f * crack_intensity
    noise = rng.normal(1.0, 0.1, size=(h, w)).astype(np.float32)
    intensity_map *= np.clip(noise, 0.7, 1.3)

    if edge_effect:
        # Distance transform from crack boundary to simulate edge diffraction
        mask_bin = (mask_f > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3).astype(np.float32)
        max_dist = dist.max() if dist.max() > 0 else 1.0
        dist_norm = dist / max_dist
        # Edge bright (dist near 0), center slightly darker
        edge_factor = 1.0 - 0.3 * dist_norm
        intensity_map *= edge_factor

    # Slight Gaussian blur to soften crack edges (depth-dependent appearance)
    intensity_map = gaussian_filter(intensity_map, sigma=0.8)

    # Overlay: cracks are brighter than background in USM
    result = result + intensity_map
    return np.clip(result, 0.0, 1.0)


def generate_synthetic_usm_with_cracks(
    height: int = 512,
    width: int = 512,
    n_cracks: Optional[int] = None,
    crack_types: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
    background_noise: float = 0.03,
    crack_intensity: float = 0.7,
    edge_effect: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a complete synthetic USM image with crack defects.

    Creates a USM-style background and overlays one or more crack patterns
    selected from the available semiconductor crack types.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    n_cracks : int, optional
        Number of crack instances (random 1..5 if ``None``).
    crack_types : list[str], optional
        Crack type names to sample from.  If ``None``, all types are used.
    rng : numpy.random.Generator, optional
        Random generator.
    background_noise : float
        USM background noise level.
    crack_intensity : float
        Crack brightness.
    edge_effect : bool
        Enable edge diffraction rendering.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        ``(image, mask, metadata)`` — float32 image in [0, 1], uint8
        binary mask, and metadata dict with crack information.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_cracks is None:
        n_cracks = int(rng.integers(1, 6))

    if crack_types is None:
        crack_types = list(_CRACK_GENERATORS.keys())

    # Generate background
    background = _generate_usm_background(height, width, rng=rng, noise_std=background_noise)

    # Accumulate crack masks
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    crack_metadata: List[dict] = []

    for i in range(n_cracks):
        ctype = str(rng.choice(crack_types))
        gen_fn = _CRACK_GENERATORS[ctype]
        crack_mask, cmeta = gen_fn(height, width, rng=rng)
        combined_mask = np.maximum(combined_mask, crack_mask)
        crack_metadata.append(cmeta)

    # Render cracks on background
    image = render_crack_on_usm(
        background, combined_mask,
        crack_intensity=crack_intensity,
        edge_effect=edge_effect,
        rng=rng,
    )

    metadata = {
        "height": height,
        "width": width,
        "n_cracks": n_cracks,
        "cracks": crack_metadata,
        "domain": "usm",
    }

    return image, combined_mask, metadata
