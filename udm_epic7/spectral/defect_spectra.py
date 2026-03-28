"""
Defect spectral signatures for Chromasense multi-spectral imaging.

Each function modifies a base material spectrum to simulate how a specific
defect type alters the spectral reflectance of a semiconductor surface.

Defect types
------------
- **Delamination**: Air gap between layers reduces reflectance and introduces
  wavelength-dependent interference fringes (simplified as wavelength-modulated
  reflectance reduction).
- **Contamination**: Foreign material deposits alter the spectrum by blending
  the contaminant's spectral signature with the base material.
- **Oxidation**: Surface oxide layer reduces metallic reflectance, particularly
  in longer wavelengths for copper-based materials.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from udm_epic7.spectral.wavelength_model import (
    SpectralConfig,
    default_spectral_config,
    material_reflectance,
)

# ---------------------------------------------------------------------------
# Contaminant spectral signatures (reflectance at default wavelengths)
# ---------------------------------------------------------------------------

_CONTAMINANT_SPECTRA: Dict[str, Dict[float, float]] = {
    "flux_residue": {
        450.0: 0.55,
        550.0: 0.60,
        650.0: 0.50,
        850.0: 0.35,
    },
    "organic_film": {
        450.0: 0.30,
        550.0: 0.35,
        650.0: 0.40,
        850.0: 0.28,
    },
    "dust": {
        450.0: 0.45,
        550.0: 0.46,
        650.0: 0.47,
        850.0: 0.48,
    },
    "solder_splash": {
        450.0: 0.40,
        550.0: 0.46,
        650.0: 0.50,
        850.0: 0.55,
    },
}


def delamination_spectrum(
    base_material: str,
    severity: float = 0.5,
    config: Optional[SpectralConfig] = None,
) -> Dict[float, float]:
    """Compute modified spectrum for a delaminated region.

    Delamination creates an air gap that reduces reflectance in a
    wavelength-dependent manner.  Short wavelengths are affected more
    due to increased scattering from the rough delamination interface.

    Parameters
    ----------
    base_material : str
        Name of the base material (e.g. ``"copper"``).
    severity : float
        Delamination severity in [0.0, 1.0].  ``0.0`` = no defect (returns
        the unmodified base spectrum); ``1.0`` = complete delamination.
    config : SpectralConfig | None
        Spectral configuration.  Defaults to standard Chromasense config.

    Returns
    -------
    dict[float, float]
        Modified reflectance per wavelength.
    """
    severity = max(0.0, min(1.0, severity))
    if config is None:
        config = default_spectral_config()

    result: Dict[float, float] = {}
    wl_min = min(config.wavelengths)
    wl_max = max(config.wavelengths)
    wl_range = wl_max - wl_min if wl_max > wl_min else 1.0

    for wl in config.wavelengths:
        base_ref = material_reflectance(base_material, wl, config)

        # Wavelength-dependent attenuation: shorter wavelengths scatter more
        # at the delamination interface.
        wl_factor = 1.0 - 0.3 * (1.0 - (wl - wl_min) / wl_range)

        # Interference fringe modulation (simplified sinusoidal)
        fringe = 0.05 * severity * math.sin(2.0 * math.pi * wl / 200.0)

        attenuation = severity * wl_factor
        modified = base_ref * (1.0 - 0.6 * attenuation) + fringe
        result[wl] = max(0.0, min(1.0, modified))

    return result


def contamination_spectrum(
    contaminant_type: str = "flux_residue",
    concentration: float = 0.5,
    base_material: str = "copper",
    config: Optional[SpectralConfig] = None,
) -> Dict[float, float]:
    """Compute spectrum for a surface contaminated by foreign material.

    The result is a weighted blend between the base material spectrum and
    the contaminant spectrum, controlled by *concentration*.

    Parameters
    ----------
    contaminant_type : str
        Type of contaminant.  Supported: ``"flux_residue"``,
        ``"organic_film"``, ``"dust"``, ``"solder_splash"``.
    concentration : float
        Contaminant concentration in [0.0, 1.0].  ``0.0`` = clean surface;
        ``1.0`` = fully covered.
    base_material : str
        Underlying material name.
    config : SpectralConfig | None
        Spectral configuration.

    Returns
    -------
    dict[float, float]
        Blended reflectance per wavelength.

    Raises
    ------
    KeyError
        If *contaminant_type* is not recognised.
    """
    concentration = max(0.0, min(1.0, concentration))
    if config is None:
        config = default_spectral_config()

    contaminant = _CONTAMINANT_SPECTRA.get(contaminant_type)
    if contaminant is None:
        raise KeyError(
            f"Unknown contaminant '{contaminant_type}'. "
            f"Available: {sorted(_CONTAMINANT_SPECTRA.keys())}"
        )

    result: Dict[float, float] = {}
    for wl in config.wavelengths:
        base_ref = material_reflectance(base_material, wl, config)
        # Linear interpolation for contaminant at non-standard wavelengths
        cont_ref = _interpolate_contaminant(contaminant, wl)
        blended = (1.0 - concentration) * base_ref + concentration * cont_ref
        result[wl] = max(0.0, min(1.0, blended))

    return result


def oxidation_spectrum(
    base_material: str = "copper",
    oxidation_level: float = 0.5,
    config: Optional[SpectralConfig] = None,
) -> Dict[float, float]:
    """Compute spectrum for an oxidised surface.

    Oxidation reduces the metallic reflectance of the base material.
    The effect is stronger in the red and near-IR bands for copper
    (CuO absorption), and roughly uniform for other materials.

    Parameters
    ----------
    base_material : str
        Name of the base material.
    oxidation_level : float
        Oxidation level in [0.0, 1.0].  ``0.0`` = pristine; ``1.0`` =
        heavily oxidised.
    config : SpectralConfig | None
        Spectral configuration.

    Returns
    -------
    dict[float, float]
        Modified reflectance per wavelength.
    """
    oxidation_level = max(0.0, min(1.0, oxidation_level))
    if config is None:
        config = default_spectral_config()

    result: Dict[float, float] = {}
    wl_min = min(config.wavelengths)
    wl_max = max(config.wavelengths)
    wl_range = wl_max - wl_min if wl_max > wl_min else 1.0

    is_copper = "copper" in base_material.lower() or "cu" in base_material.lower()

    for wl in config.wavelengths:
        base_ref = material_reflectance(base_material, wl, config)

        if is_copper:
            # Copper oxide absorbs more strongly in the red/IR
            wl_normalized = (wl - wl_min) / wl_range  # 0 at blue, 1 at IR
            reduction = oxidation_level * (0.3 + 0.4 * wl_normalized)
        else:
            # Uniform reduction for non-copper materials
            reduction = oxidation_level * 0.35

        modified = base_ref * (1.0 - reduction)
        result[wl] = max(0.0, min(1.0, modified))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interpolate_contaminant(spectra: Dict[float, float], wavelength: float) -> float:
    """Linearly interpolate contaminant reflectance at an arbitrary wavelength."""
    wl_sorted = sorted(spectra.keys())
    ref_sorted = [spectra[w] for w in wl_sorted]

    if wavelength in spectra:
        return spectra[wavelength]
    if wavelength <= wl_sorted[0]:
        return ref_sorted[0]
    if wavelength >= wl_sorted[-1]:
        return ref_sorted[-1]

    for i in range(len(wl_sorted) - 1):
        if wl_sorted[i] <= wavelength <= wl_sorted[i + 1]:
            t = (wavelength - wl_sorted[i]) / (wl_sorted[i + 1] - wl_sorted[i])
            return ref_sorted[i] + t * (ref_sorted[i + 1] - ref_sorted[i])

    return ref_sorted[0]
