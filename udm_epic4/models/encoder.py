"""
Epic 4 — Shared feature encoder backed by timm.

Extracts multi-scale features from 4 backbone stages for U-Net skip
connections and domain-adversarial training.  Any timm backbone that
supports ``features_only=True`` can be swapped in via *backbone_name*.
"""

from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    Multi-scale feature extractor using a timm backbone.

    Produces feature maps at 4 spatial resolutions (typically 1/4, 1/8,
    1/16, 1/32 of the input) that feed both the segmentation decoder and
    the domain classifier.

    Args:
        backbone_name: Any timm model name compatible with ``features_only``
                       (default ``"convnext_tiny"``).
        pretrained:    Load ImageNet-pretrained weights.
        in_chans:      Number of input channels (1 for grayscale X-ray,
                       3 for RGB).
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=in_chans,
        )
        # Cache channel counts so downstream modules can query them.
        self._feature_channels: List[int] = self.backbone.feature_info.channels()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_channels(self) -> List[int]:
        """Channel count for each of the 4 output stages."""
        return list(self._feature_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor of shape ``[B, C, H, W]``.

        Returns:
            List of 4 feature tensors, from highest to lowest resolution.
        """
        return self.backbone(x)
