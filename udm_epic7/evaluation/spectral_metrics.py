"""
Spectral evaluation metrics for Chromasense multi-spectral imaging.

Provides the Spectral Angle Mapper (SAM) for comparing spectral signatures
and per-pixel anomaly scoring against a reference material spectrum.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from udm_epic7.spectral.wavelength_model import (
    SpectralConfig,
    default_spectral_config,
    material_reflectance,
)

logger = logging.getLogger(__name__)


def spectral_angle_mapper(
    pred_spectrum: np.ndarray,
    true_spectrum: np.ndarray,
) -> float:
    """Compute the Spectral Angle Mapper (SAM) between two spectra.

    SAM measures the angle (in radians) between two spectral vectors,
    ignoring intensity differences.  A SAM of 0.0 means the spectra are
    identical in shape; pi/2 means they are orthogonal.

    .. math::
        \\text{SAM} = \\arccos\\left(
            \\frac{\\mathbf{a} \\cdot \\mathbf{b}}
                 {\\|\\mathbf{a}\\| \\, \\|\\mathbf{b}\\|}
        \\right)

    Parameters
    ----------
    pred_spectrum : np.ndarray
        Predicted spectral vector of shape ``(C,)`` where C is the
        number of wavelength channels.
    true_spectrum : np.ndarray
        Ground-truth spectral vector of shape ``(C,)``.

    Returns
    -------
    float
        Spectral angle in radians, in [0, pi/2].
    """
    pred = np.asarray(pred_spectrum, dtype=np.float64).ravel()
    true = np.asarray(true_spectrum, dtype=np.float64).ravel()

    norm_pred = np.linalg.norm(pred)
    norm_true = np.linalg.norm(true)

    if norm_pred == 0.0 or norm_true == 0.0:
        return 0.0

    cos_angle = np.dot(pred, true) / (norm_pred * norm_true)
    # Clamp to handle numerical precision issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def spectral_anomaly_score(
    image: np.ndarray,
    reference_spectrum: np.ndarray,
    config: Optional[SpectralConfig] = None,
) -> np.ndarray:
    """Compute per-pixel spectral anomaly score against a reference spectrum.

    For each pixel, the SAM between its spectral vector and the reference
    is computed.  Pixels that deviate significantly from the reference
    material will have higher anomaly scores.

    Parameters
    ----------
    image : np.ndarray
        Multi-spectral image of shape ``[C, H, W]``.
    reference_spectrum : np.ndarray
        Reference spectral vector of shape ``(C,)`` representing the
        expected (defect-free) material signature.
    config : SpectralConfig | None
        Spectral configuration (used for logging/metadata only).

    Returns
    -------
    np.ndarray
        Float32 anomaly score map of shape ``(H, W)`` with values in
        [0, pi/2].  Higher values indicate stronger anomaly.
    """
    if config is None:
        config = default_spectral_config()

    c, h, w = image.shape
    ref = np.asarray(reference_spectrum, dtype=np.float64).ravel()

    if ref.shape[0] != c:
        raise ValueError(
            f"Reference spectrum has {ref.shape[0]} channels but image has {c}."
        )

    # Flatten spatial dimensions for vectorised computation
    pixels = image.reshape(c, -1).T.astype(np.float64)  # (H*W, C)

    # Compute norms
    pixel_norms = np.linalg.norm(pixels, axis=1)  # (H*W,)
    ref_norm = np.linalg.norm(ref)

    # Avoid division by zero
    valid = (pixel_norms > 0) & (ref_norm > 0)
    scores = np.zeros(pixels.shape[0], dtype=np.float64)

    if valid.any():
        cos_angles = np.dot(pixels[valid], ref) / (pixel_norms[valid] * ref_norm)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        scores[valid] = np.arccos(cos_angles)

    return scores.reshape(h, w).astype(np.float32)


def spectral_anomaly_score_from_material(
    image: np.ndarray,
    material: str,
    config: Optional[SpectralConfig] = None,
) -> np.ndarray:
    """Convenience wrapper: compute anomaly score against a known material.

    Parameters
    ----------
    image : np.ndarray
        Multi-spectral image ``[C, H, W]``.
    material : str
        Reference material name (e.g. ``"copper"``).
    config : SpectralConfig | None
        Spectral configuration.

    Returns
    -------
    np.ndarray
        Anomaly score map ``(H, W)``.
    """
    if config is None:
        config = default_spectral_config()

    ref = np.array(
        [material_reflectance(material, wl, config) for wl in config.wavelengths],
        dtype=np.float64,
    )
    return spectral_anomaly_score(image, ref, config)
