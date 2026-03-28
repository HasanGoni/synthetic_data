"""
UDM Epic 4 — DANN model components for domain-adversarial void segmentation.
"""

from udm_epic4.models.encoder import SharedEncoder
from udm_epic4.models.decoder import UNetDecoder, DecoderBlock
from udm_epic4.models.domain_classifier import (
    DomainClassifier,
    GradientReversalLayer,
    GradientReversalFunction,
)
from udm_epic4.models.dann import DANNModel

__all__ = [
    "SharedEncoder",
    "UNetDecoder",
    "DecoderBlock",
    "DomainClassifier",
    "GradientReversalLayer",
    "GradientReversalFunction",
    "DANNModel",
]
