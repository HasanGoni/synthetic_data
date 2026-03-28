"""
Multi-spectral image renderer for Chromasense synthetic data.

Renders synthetic multi-channel spectral images of semiconductor packages
with optional defect regions.  Each channel corresponds to a Chromasense
illumination wavelength and encodes the material-specific reflectance at
that wavelength, plus sensor noise and spatial texture.

Layout format
-------------
A *layout* is a 2-D ``np.ndarray`` of dtype ``int`` (or ``str`` via dict),
where each integer labels a distinct material region:

    0 = background / mold_compound
    1 = silicon
    2 = copper
    3 = solder

A *defects* list contains dicts describing defect overlays::

    [
        {
            "type": "delamination" | "contamination" | "oxidation",
            "bbox": (y0, x0, y1, x1),   # region of interest
            "severity": 0.0 .. 1.0,
            "material": "copper",        # affected base material
            "contaminant": "flux_residue", # only for contamination
        },
        ...
    ]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from udm_epic7.spectral.wavelength_model import (
    SpectralConfig,
    default_spectral_config,
    material_reflectance,
)
from udm_epic7.spectral.defect_spectra import (
    contamination_spectrum,
    delamination_spectrum,
    oxidation_spectrum,
)

logger = logging.getLogger(__name__)

# Default mapping from integer layout labels to material names.
_LABEL_TO_MATERIAL: Dict[int, str] = {
    0: "mold_compound",
    1: "silicon",
    2: "copper",
    3: "solder",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_spectral_image(
    layout: np.ndarray,
    defects: Optional[List[dict]] = None,
    config: Optional[SpectralConfig] = None,
    height: int = 512,
    width: int = 512,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.02,
    label_to_material: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Render a multi-channel spectral image.

    Parameters
    ----------
    layout : np.ndarray
        Integer material-label map of shape ``(H_layout, W_layout)``.
        Resized to ``(height, width)`` if dimensions differ.
    defects : list[dict] | None
        Defect descriptors (see module docstring for format).
    config : SpectralConfig | None
        Spectral configuration.
    height, width : int
        Output spatial dimensions.
    rng : np.random.Generator | None
        Random number generator for reproducible noise.
    noise_std : float
        Standard deviation of additive Gaussian sensor noise.
    label_to_material : dict[int, str] | None
        Override mapping from layout integer labels to material names.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[C, H, W]`` where ``C`` is the number
        of wavelengths.  Values are in [0, 1].
    """
    if config is None:
        config = default_spectral_config()
    if rng is None:
        rng = np.random.default_rng()
    if label_to_material is None:
        label_to_material = _LABEL_TO_MATERIAL
    if defects is None:
        defects = []

    n_ch = config.n_channels

    # Resize layout if needed
    layout_resized = _resize_label_map(layout, height, width)

    # Render each wavelength channel
    channels = []
    for wl in config.wavelengths:
        ch = render_single_wavelength(
            layout_resized, wl, config,
            height=height, width=width, rng=rng,
            noise_std=noise_std, label_to_material=label_to_material,
        )
        channels.append(ch)

    image = np.stack(channels, axis=0)  # [C, H, W]

    # Apply defect overlays
    for defect in defects:
        image = _apply_defect_overlay(image, defect, config, label_to_material)

    # Clamp to [0, 1]
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def render_single_wavelength(
    layout: np.ndarray,
    wavelength: float,
    config: Optional[SpectralConfig] = None,
    height: int = 512,
    width: int = 512,
    rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.02,
    label_to_material: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Render a single-channel image for one wavelength.

    Parameters
    ----------
    layout : np.ndarray
        Integer material-label map, shape ``(H, W)``.
    wavelength : float
        Illumination wavelength in nm.
    config : SpectralConfig | None
        Spectral configuration.
    height, width : int
        Output spatial dimensions.
    rng : np.random.Generator | None
        Random number generator.
    noise_std : float
        Additive Gaussian noise standard deviation.
    label_to_material : dict[int, str] | None
        Label-to-material mapping.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(H, W)`` with values in approximately [0, 1].
    """
    if config is None:
        config = default_spectral_config()
    if rng is None:
        rng = np.random.default_rng()
    if label_to_material is None:
        label_to_material = _LABEL_TO_MATERIAL

    layout_resized = _resize_label_map(layout, height, width)

    # Build reflectance map
    channel = np.zeros((height, width), dtype=np.float32)
    unique_labels = np.unique(layout_resized)

    for label_val in unique_labels:
        mat_name = label_to_material.get(int(label_val), "mold_compound")
        ref = material_reflectance(mat_name, wavelength, config)
        mask = layout_resized == label_val
        # Add slight spatial texture (Perlin-like low-freq noise)
        texture = _spatial_texture(height, width, rng, scale=0.03)
        channel[mask] = ref + texture[mask]

    # Additive sensor noise
    if noise_std > 0:
        channel += rng.normal(0.0, noise_std, size=(height, width)).astype(np.float32)

    return channel


def spectral_to_rgb(
    spectral_image: np.ndarray,
    config: Optional[SpectralConfig] = None,
) -> np.ndarray:
    """Convert a multi-spectral image to an approximate RGB visualisation.

    Uses simplified CIE-like wavelength-to-RGB mapping.  Channels outside
    the visible range (> 700 nm) are mapped to a dim red for visibility.

    Parameters
    ----------
    spectral_image : np.ndarray
        Float32 array of shape ``[C, H, W]``.
    config : SpectralConfig | None
        Spectral configuration.

    Returns
    -------
    np.ndarray
        Uint8 RGB image of shape ``[H, W, 3]``.
    """
    if config is None:
        config = default_spectral_config()

    c, h, w = spectral_image.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for i, wl in enumerate(config.wavelengths):
        r, g, b = _wavelength_to_rgb_weights(wl)
        rgb[:, :, 0] += spectral_image[i] * r
        rgb[:, :, 1] += spectral_image[i] * g
        rgb[:, :, 2] += spectral_image[i] * b

    # Normalise to [0, 255]
    rgb_max = rgb.max()
    if rgb_max > 0:
        rgb = rgb / rgb_max
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_label_map(layout: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an integer label map using nearest-neighbour interpolation."""
    if layout.shape[0] == height and layout.shape[1] == width:
        return layout

    import cv2
    return cv2.resize(
        layout.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(layout.dtype)


def _spatial_texture(
    height: int, width: int, rng: np.random.Generator, scale: float = 0.03
) -> np.ndarray:
    """Generate low-frequency spatial texture to simulate surface variation."""
    from scipy.ndimage import gaussian_filter

    noise = rng.normal(0.0, scale, size=(height, width)).astype(np.float32)
    # Smooth to create low-frequency spatial variation
    smoothed = gaussian_filter(noise, sigma=max(height, width) / 32.0)
    return smoothed


def _apply_defect_overlay(
    image: np.ndarray,
    defect: dict,
    config: SpectralConfig,
    label_to_material: Dict[int, str],
) -> np.ndarray:
    """Apply a single defect overlay to the multi-channel image in-place."""
    dtype = defect.get("type", "delamination")
    severity = defect.get("severity", 0.5)
    material = defect.get("material", "copper")
    bbox = defect.get("bbox", (0, 0, image.shape[1], image.shape[2]))
    y0, x0, y1, x1 = bbox
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(image.shape[1], y1), min(image.shape[2], x1)

    if dtype == "delamination":
        defect_spec = delamination_spectrum(material, severity, config)
    elif dtype == "contamination":
        contaminant = defect.get("contaminant", "flux_residue")
        defect_spec = contamination_spectrum(contaminant, severity, material, config)
    elif dtype == "oxidation":
        defect_spec = oxidation_spectrum(material, severity, config)
    else:
        logger.warning("Unknown defect type '%s', skipping", dtype)
        return image

    # Overwrite the defect bbox region with the defect spectrum
    for ch_idx, wl in enumerate(config.wavelengths):
        defect_ref = defect_spec.get(wl, 0.0)
        # Blend: replace interior with defect reflectance
        image[ch_idx, y0:y1, x0:x1] = defect_ref

    return image


def _wavelength_to_rgb_weights(wavelength: float) -> tuple:
    """Map a wavelength (nm) to approximate (R, G, B) weights.

    Based on a simplified version of CIE colour matching.  Near-IR
    wavelengths (> 700 nm) are mapped to a dim red for visualisation.
    """
    wl = wavelength
    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    elif wl <= 700:
        r, g, b = 1.0, 0.0, 0.0
    else:
        # Near-IR: dim red channel
        r, g, b = 0.3, 0.0, 0.0

    return max(0.0, r), max(0.0, g), max(0.0, b)
