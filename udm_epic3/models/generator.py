"""
Epic 3 — ResNet-based generator for CycleGAN image-to-image translation.

Implements a ResNet generator architecture following Johnson et al. (2016) with
reflection padding and instance normalisation, suitable for unpaired
cross-modality translation between AOI and USM inspection images.

Usage::

    from udm_epic3.models.generator import ResnetGenerator

    G = ResnetGenerator(in_channels=1, out_channels=1, n_filters=64, n_blocks=9)
    fake_B = G(real_A)  # [B, 1, H, W] -> [B, 1, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """
    Single ResNet block with two 3x3 convolutions, instance normalisation,
    and a skip (identity) connection.

    Args:
        dim: Number of input and output channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block: ``out = x + block(x)``."""
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based generator for image-to-image translation.

    Architecture overview::

        ReflectionPad → Conv7 → InstanceNorm → ReLU
        → 2 × (Conv3 stride-2 downsample)
        → N × ResnetBlock
        → 2 × (ConvTranspose3 stride-2 upsample)
        → ReflectionPad → Conv7 → Tanh

    All normalisation layers use ``InstanceNorm2d`` and reflection padding is
    used throughout to reduce boundary artefacts.

    Args:
        in_channels:  Number of input image channels (1 for grayscale).
        out_channels: Number of output image channels (1 for grayscale).
        n_filters:    Base number of convolutional filters.
        n_blocks:     Number of ResNet blocks in the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_filters: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()

        # -- Initial convolution block: Conv7 → InstanceNorm → ReLU ---------
        initial = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_filters, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(inplace=True),
        ]

        # -- Downsampling: 2 stride-2 convolutions --------------------------
        downsampling: list[nn.Module] = []
        in_f = n_filters
        for i in range(2):
            out_f = in_f * 2
            downsampling += [
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # -- ResNet bottleneck blocks ----------------------------------------
        resnet_blocks: list[nn.Module] = []
        for _ in range(n_blocks):
            resnet_blocks.append(ResnetBlock(in_f))

        # -- Upsampling: 2 fractional-stride convolutions --------------------
        upsampling: list[nn.Module] = []
        for i in range(2):
            out_f = in_f // 2
            upsampling += [
                nn.ConvTranspose2d(
                    in_f, out_f,
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # -- Final convolution block: Conv7 → Tanh --------------------------
        output = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(
            *initial, *downsampling, *resnet_blocks, *upsampling, *output,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Translate an input image to the target domain.

        Args:
            x: Input tensor of shape ``[B, C_in, H, W]``.

        Returns:
            Translated image tensor of shape ``[B, C_out, H, W]`` in ``[-1, 1]``.
        """
        return self.model(x)
