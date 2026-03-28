"""
Epic 4 — Domain-Adversarial Neural Network (DANN) for void segmentation.

Composes the shared encoder, U-Net decoder, and domain classifier into a
single model that jointly optimises segmentation accuracy and
domain-invariant feature learning.

Usage::

    from udm_epic4.models import DANNModel

    model = DANNModel(backbone="convnext_tiny", pretrained=True)
    seg_logits, domain_logits = model(images, lambda_val=0.5)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udm_epic4.models.encoder import SharedEncoder
from udm_epic4.models.decoder import UNetDecoder
from udm_epic4.models.domain_classifier import DomainClassifier


class DANNModel(nn.Module):
    """
    Full DANN: SharedEncoder + UNetDecoder + DomainClassifier.

    The encoder produces multi-scale features.  The deepest features
    (bottleneck) are simultaneously fed to the segmentation decoder
    (via skip-connected upsampling) and to the domain classifier
    (via gradient reversal).

    Args:
        backbone:          timm backbone name (default ``"convnext_tiny"``).
        pretrained:        Load ImageNet-pretrained encoder weights.
        in_chans:          Number of input image channels.
        decoder_channels:  Channel widths for each decoder stage.
        domain_head_hidden: Hidden-layer width in the domain classifier.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_channels: List[int] | None = None,
        domain_head_hidden: int = 256,
    ) -> None:
        super().__init__()
        decoder_channels = decoder_channels or [256, 128, 64, 32]

        self.encoder = SharedEncoder(
            backbone_name=backbone,
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=decoder_channels,
        )
        # Domain classifier operates on the bottleneck (deepest) features.
        bottleneck_channels = self.encoder.feature_channels[-1]
        self.domain_classifier = DomainClassifier(
            feature_dim=bottleneck_channels,
            hidden_dim=domain_head_hidden,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lambda_val: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: segmentation + domain classification.

        Args:
            x:          Input images ``[B, C, H, W]``.
            lambda_val: GRL strength (anneal from 0 -> 1 during training).

        Returns:
            seg_logits:    Segmentation logits ``[B, 1, H, W]`` (may need
                           interpolation to match the exact input size).
            domain_logits: Domain logits ``[B, 1]``.
        """
        features: List[torch.Tensor] = self.encoder(x)

        # Segmentation path
        seg_logits = self.decoder(features)
        # Upsample to input resolution if the decoder output is smaller.
        if seg_logits.shape[2:] != x.shape[2:]:
            seg_logits = F.interpolate(
                seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False,
            )

        # Domain classification on bottleneck features
        bottleneck = features[-1]
        domain_logits = self.domain_classifier(bottleneck, lambda_val=lambda_val)

        return seg_logits, domain_logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract bottleneck features (useful for t-SNE / embedding analysis).

        Args:
            x: Input images ``[B, C, H, W]``.

        Returns:
            Bottleneck feature map ``[B, C_bottleneck, H', W']``.
        """
        features = self.encoder(x)
        return features[-1]
