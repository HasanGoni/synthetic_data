"""
Epic 9 — Domain transfer from USM to RGB and mask-to-image generation.

Provides two main capabilities:

1. :class:`USMtoRGBTransfer` — converts grayscale USM images into
   pseudo-RGB representations using physically-motivated colormaps
   (with a placeholder path for learned CycleGAN transfer).

2. :func:`mask_to_image` — conditioned image generation: given *only*
   a binary crack mask, produces a full synthetic image in the specified
   target domain (USM or RGB).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from udm_epic9.rendering.usm_renderer import (
    _generate_usm_background,
    render_crack_on_usm,
)

logger = logging.getLogger(__name__)

# OpenCV colormap constants indexed by name
_COLORMAP_LUT = {
    "inferno": cv2.COLORMAP_INFERNO,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


class USMtoRGBTransfer:
    """Transfer USM grayscale images to pseudo-RGB domain.

    Parameters
    ----------
    method : str
        Transfer method — ``'colormap'`` for physically-motivated
        pseudo-colour mapping, or ``'learned'`` (placeholder for a
        CycleGAN-based transfer that can be trained separately).
    colormap : str
        Colormap name when using the ``'colormap'`` method.  Supported
        values: ``inferno``, ``jet``, ``hot``, ``magma``, ``plasma``,
        ``turbo``, ``viridis``.
    """

    def __init__(self, method: str = "colormap", colormap: str = "inferno") -> None:
        if method not in ("colormap", "learned"):
            raise ValueError(f"Unknown method '{method}', expected 'colormap' or 'learned'")
        self.method = method
        self.colormap = colormap
        self._cv_cmap = _COLORMAP_LUT.get(colormap, cv2.COLORMAP_INFERNO)

    # ── Public API ─────────────────────────────────────────────────────

    def transfer(self, usm_image: np.ndarray) -> np.ndarray:
        """Convert a grayscale USM image to pseudo-RGB.

        Parameters
        ----------
        usm_image : np.ndarray
            Grayscale float32 image in [0, 1], shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            RGB uint8 image of shape ``(H, W, 3)``.
        """
        if self.method == "colormap":
            return self._transfer_colormap(usm_image)
        else:
            return self._transfer_learned(usm_image)

    def transfer_with_cracks(
        self,
        usm_image: np.ndarray,
        crack_mask: np.ndarray,
    ) -> np.ndarray:
        """Transfer USM to RGB while preserving crack visibility.

        Applies the domain transfer and then boosts the crack region
        contrast so cracks remain clearly visible in the RGB output.

        Parameters
        ----------
        usm_image : np.ndarray
            Grayscale float32 image in [0, 1].
        crack_mask : np.ndarray
            Binary crack mask (uint8 0/255 or float 0/1).

        Returns
        -------
        np.ndarray
            RGB uint8 image of shape ``(H, W, 3)``.
        """
        rgb = self.transfer(usm_image)

        # Normalise mask
        mask_f = crack_mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        # Boost crack region brightness in RGB
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        boost = 40.0  # additive brightness boost
        rgb_out = rgb.astype(np.float32) + mask_3ch * boost
        return np.clip(rgb_out, 0, 255).astype(np.uint8)

    # ── Private methods ────────────────────────────────────────────────

    def _transfer_colormap(self, usm_image: np.ndarray) -> np.ndarray:
        """Apply OpenCV colormap to produce pseudo-RGB."""
        # Scale to uint8
        gray_u8 = np.clip(usm_image * 255.0, 0, 255).astype(np.uint8)
        # Apply colormap (returns BGR)
        bgr = cv2.applyColorMap(gray_u8, self._cv_cmap)
        # Convert to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _transfer_learned(self, usm_image: np.ndarray) -> np.ndarray:
        """Placeholder for learned (CycleGAN) domain transfer.

        Falls back to colormap transfer with a warning.  To enable
        learned transfer, subclass and override this method with a
        trained generator network.
        """
        logger.warning(
            "Learned (CycleGAN) transfer not yet trained — "
            "falling back to colormap transfer.  Train a CycleGAN "
            "on paired USM/RGB data and override _transfer_learned()."
        )
        return self._transfer_colormap(usm_image)


# ── Mask-to-image generation ──────────────────────────────────────────────


def mask_to_image(
    crack_mask: np.ndarray,
    target_domain: str = "usm",
    height: int = 512,
    width: int = 512,
    rng: Optional[np.random.Generator] = None,
    crack_intensity: float = 0.7,
    colormap: str = "inferno",
) -> np.ndarray:
    """Generate a full synthetic image from only a crack mask.

    This is the **mask-to-image** capability: given a binary crack mask,
    produce a realistic image in the specified target domain.

    - For ``usm``: generates a USM background and overlays the crack.
    - For ``rgb``: generates USM first, then applies domain transfer.

    Parameters
    ----------
    crack_mask : np.ndarray
        Binary crack mask, uint8 shape ``(H, W)``.
    target_domain : str
        Target domain — ``'usm'`` or ``'rgb'``.
    height, width : int
        Image dimensions (used if mask needs resizing).
    rng : numpy.random.Generator, optional
        Random generator.
    crack_intensity : float
        Crack brightness for USM rendering.
    colormap : str
        Colormap for RGB domain transfer.

    Returns
    -------
    np.ndarray
        Generated image — float32 ``(H, W)`` for USM, uint8
        ``(H, W, 3)`` for RGB.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = crack_mask.shape[:2]
    # Use mask dimensions if they differ from defaults
    if height == 512 and width == 512 and (h != height or w != width):
        height, width = h, w
    if h != height or w != width:
        crack_mask = cv2.resize(crack_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        h, w = height, width

    # Generate USM background
    background = _generate_usm_background(h, w, rng=rng)

    # Render crack on USM
    usm_image = render_crack_on_usm(
        background, crack_mask,
        crack_intensity=crack_intensity,
        edge_effect=True,
        rng=rng,
    )

    if target_domain == "usm":
        return usm_image

    # Transfer to RGB
    transfer = USMtoRGBTransfer(method="colormap", colormap=colormap)
    rgb_image = transfer.transfer_with_cracks(usm_image, crack_mask)
    return rgb_image
