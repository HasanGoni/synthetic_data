"""
Epic 3 — PatchGAN discriminator for CycleGAN.

Implements a 70x70 PatchGAN discriminator that classifies overlapping image
patches as real or fake, following Isola et al. (2017).  The output is *not*
passed through a sigmoid — raw logits are returned for use with the LSGAN
least-squares objective.

Usage::

    from udm_epic3.models.discriminator import PatchDiscriminator

    D = PatchDiscriminator(in_channels=1, n_filters=64)
    patch_scores = D(image)  # [B, 1, H', W']
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Architecture overview::

        Conv4 stride-2 → LeakyReLU                     (64)
        Conv4 stride-2 → InstanceNorm → LeakyReLU      (128)
        Conv4 stride-2 → InstanceNorm → LeakyReLU      (256)
        Conv4 stride-1 → InstanceNorm → LeakyReLU      (512)
        Conv4 stride-1 → 1                              (1)

    The first layer omits instance normalisation.  No sigmoid is applied to
    the output so that the LSGAN (least-squares) loss can be used directly.

    Args:
        in_channels: Number of input image channels (1 for grayscale).
        n_filters:   Base number of convolutional filters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters: int = 64,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            # Layer 1 — no InstanceNorm
            nn.Conv2d(
                in_channels, n_filters,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layers 2-4: increasing filters with InstanceNorm
        filter_mult_prev = 1
        for n in range(1, 4):
            filter_mult = min(2 ** n, 8)
            stride = 2 if n < 3 else 1  # stride-2 for layers 2-3, stride-1 for layer 4
            layers += [
                nn.Conv2d(
                    n_filters * filter_mult_prev,
                    n_filters * filter_mult,
                    kernel_size=4, stride=stride, padding=1, bias=False,
                ),
                nn.InstanceNorm2d(n_filters * filter_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            filter_mult_prev = filter_mult

        # Final prediction layer — single channel output, no activation
        layers.append(
            nn.Conv2d(
                n_filters * filter_mult_prev, 1,
                kernel_size=4, stride=1, padding=1,
            ),
        )

        self.model = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-level real/fake scores.

        Args:
            x: Input image tensor of shape ``[B, C_in, H, W]``.

        Returns:
            Patch prediction map of shape ``[B, 1, H', W']`` (raw logits,
            no sigmoid).
        """
        return self.model(x)
