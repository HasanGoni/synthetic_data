"""
UDM Epic 7 — Chromasense Multi-Spectral Integration.

Generate synthetic multi-spectral images for semiconductor material defect
detection.  Chromasense uses multiple wavelengths (visible + near-IR) to
detect material composition defects (delamination, contamination, oxidation)
that are invisible in standard single-band imaging.

Install extras: ``pip install -e ".[epic7]"``
"""

from udm_epic7.spectral.wavelength_model import (
    SpectralConfig,
    default_spectral_config,
    material_reflectance,
)
from udm_epic7.spectral.defect_spectra import (
    delamination_spectrum,
    contamination_spectrum,
    oxidation_spectrum,
)

__all__ = [
    "SpectralConfig",
    "default_spectral_config",
    "material_reflectance",
    "delamination_spectrum",
    "contamination_spectrum",
    "oxidation_spectrum",
]
