"""
UDM Epic 5 — Uncertainty quantification for active domain adaptation.
"""

from udm_epic5.uncertainty.mc_dropout import (
    mc_dropout_uncertainty,
    compute_entropy,
)

__all__ = [
    "mc_dropout_uncertainty",
    "compute_entropy",
]
