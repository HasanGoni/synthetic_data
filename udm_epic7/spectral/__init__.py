"""Spectral modelling: wavelength configurations, material reflectance, defect spectra."""

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
