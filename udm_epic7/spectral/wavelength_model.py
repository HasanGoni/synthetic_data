"""
Spectral configuration and material reflectance model.

Defines the Chromasense wavelength set and per-material reflectance curves
used to render synthetic multi-spectral images of semiconductor packages.

Materials
---------
- **silicon (Si)**: High reflectance in near-IR, moderate in visible.
- **copper (Cu)**: Low blue reflectance, high red/IR (characteristic copper colour).
- **solder (SnAgCu)**: Relatively flat, mildly increasing toward IR.
- **mold_compound**: Dark epoxy; low, roughly flat reflectance.
- **oxidized_copper (CuO/Cu2O)**: Reduced reflectance compared to clean Cu,
  especially in the red/IR region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default material reflectance look-up (wavelength in nm -> reflectance 0-1)
# Values are representative of semiconductor packaging materials measured
# under Chromasense-class multi-spectral illumination.
# ---------------------------------------------------------------------------

_DEFAULT_WAVELENGTHS: List[float] = [450.0, 550.0, 650.0, 850.0]

_DEFAULT_MATERIAL_SPECTRA: Dict[str, Dict[float, float]] = {
    "silicon": {
        450.0: 0.38,
        550.0: 0.35,
        650.0: 0.34,
        850.0: 0.33,
    },
    "copper": {
        450.0: 0.18,
        550.0: 0.45,
        650.0: 0.72,
        850.0: 0.83,
    },
    "solder": {
        450.0: 0.42,
        550.0: 0.48,
        650.0: 0.52,
        850.0: 0.58,
    },
    "mold_compound": {
        450.0: 0.08,
        550.0: 0.09,
        650.0: 0.10,
        850.0: 0.12,
    },
    "oxidized_copper": {
        450.0: 0.12,
        550.0: 0.22,
        650.0: 0.35,
        850.0: 0.42,
    },
}


@dataclass
class SpectralConfig:
    """Configuration for a Chromasense multi-spectral imaging session.

    Parameters
    ----------
    wavelengths : list[float]
        Illumination wavelengths in nanometres.
    material_spectra : dict[str, dict[float, float]]
        Mapping *material_name* -> {wavelength_nm: reflectance}.
        Reflectance values are in [0, 1].
    """

    wavelengths: List[float] = field(default_factory=lambda: list(_DEFAULT_WAVELENGTHS))
    material_spectra: Dict[str, Dict[float, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_MATERIAL_SPECTRA.items()
        }
    )

    @property
    def n_channels(self) -> int:
        """Number of spectral channels (one per wavelength)."""
        return len(self.wavelengths)

    @property
    def material_names(self) -> List[str]:
        """Sorted list of known material names."""
        return sorted(self.material_spectra.keys())


def default_spectral_config() -> SpectralConfig:
    """Return a :class:`SpectralConfig` with standard Chromasense wavelengths.

    Default wavelengths: 450 nm (blue), 550 nm (green), 650 nm (red),
    850 nm (near-IR).
    """
    return SpectralConfig()


def material_reflectance(
    material: str,
    wavelength: float,
    config: Optional[SpectralConfig] = None,
) -> float:
    """Look up (or linearly interpolate) reflectance for *material* at *wavelength*.

    If the exact wavelength is present in the config it is returned directly.
    Otherwise, the two nearest calibrated wavelengths are used for linear
    interpolation.  Queries outside the calibrated range are clamped to the
    nearest endpoint.

    Parameters
    ----------
    material : str
        Material name (must exist in *config.material_spectra*).
    wavelength : float
        Query wavelength in nanometres.
    config : SpectralConfig | None
        Spectral configuration.  ``None`` uses :func:`default_spectral_config`.

    Returns
    -------
    float
        Reflectance in [0, 1].

    Raises
    ------
    KeyError
        If *material* is not found in the configuration.
    """
    if config is None:
        config = default_spectral_config()

    spectra = config.material_spectra.get(material)
    if spectra is None:
        raise KeyError(
            f"Unknown material '{material}'. "
            f"Available: {sorted(config.material_spectra.keys())}"
        )

    # Sort calibration points by wavelength
    wl_sorted = sorted(spectra.keys())
    ref_sorted = [spectra[w] for w in wl_sorted]

    # Exact match
    if wavelength in spectra:
        return spectra[wavelength]

    # Clamp to endpoints
    if wavelength <= wl_sorted[0]:
        return ref_sorted[0]
    if wavelength >= wl_sorted[-1]:
        return ref_sorted[-1]

    # Linear interpolation between the two bracketing wavelengths
    for i in range(len(wl_sorted) - 1):
        if wl_sorted[i] <= wavelength <= wl_sorted[i + 1]:
            t = (wavelength - wl_sorted[i]) / (wl_sorted[i + 1] - wl_sorted[i])
            return ref_sorted[i] + t * (ref_sorted[i + 1] - ref_sorted[i])

    # Should not reach here, but fall back to nearest
    idx = int(np.argmin(np.abs(np.array(wl_sorted) - wavelength)))
    return ref_sorted[idx]
