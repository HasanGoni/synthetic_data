"""Multi-spectral rendering: image synthesis and RGB conversion."""

from udm_epic7.rendering.spectral_renderer import (
    render_spectral_image,
    render_single_wavelength,
    spectral_to_rgb,
)

__all__ = [
    "render_spectral_image",
    "render_single_wavelength",
    "spectral_to_rgb",
]
