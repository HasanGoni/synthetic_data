"""
Epic 4 — U-Net segmentation decoder.

Progressively upsamples and fuses multi-scale encoder features back to
the original spatial resolution, producing a single-channel segmentation
logit map suitable for binary void detection.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ── Building blocks ──────────────────────────────────────────────────────────


class DecoderBlock(nn.Module):
    """
    Single U-Net decoder stage.

    ``ConvTranspose2d`` doubles the spatial dims, the result is concatenated
    with the corresponding skip connection, then refined with two
    ``Conv-BN-ReLU`` layers.

    Args:
        in_channels:   Channels coming from the previous (deeper) decoder
                       stage.
        skip_channels: Channels in the encoder skip connection at this level.
        out_channels:  Output channels after the two conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2,
        )
        # After concat: out_channels (from upsample) + skip_channels
        cat_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(cat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    Feature map from the deeper decoder level ``[B, C_in, H, W]``.
            skip: Encoder skip connection ``[B, C_skip, 2H, 2W]``.

        Returns:
            Refined feature map ``[B, C_out, 2H, 2W]``.
        """
        x = self.upsample(x)
        # Handle slight spatial mismatches caused by odd input dims.
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ── Full decoder ─────────────────────────────────────────────────────────────


class UNetDecoder(nn.Module):
    """
    U-Net decoder that fuses 4 encoder feature stages into a
    single-channel segmentation logit map.

    The decoder is built bottom-up: the deepest encoder features are
    progressively upsampled and concatenated with skip connections from
    shallower stages.

    Args:
        encoder_channels: Channel counts for the 4 encoder stages,
                          ordered from shallowest to deepest
                          (e.g. ``[96, 192, 384, 768]`` for ConvNeXt-Tiny).
        decoder_channels: Output channels for each decoder block
                          (default ``[256, 128, 64, 32]``).
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] | None = None,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError(
                f"Expected 4 encoder stages, got {len(encoder_channels)}"
            )
        decoder_channels = decoder_channels or [256, 128, 64, 32]
        if len(decoder_channels) != 4:
            raise ValueError(
                f"Expected 4 decoder channel values, got {len(decoder_channels)}"
            )

        # encoder_channels: [s0, s1, s2, s3]  (shallow → deep)
        # Decoder block 0: in=s3,           skip=s2, out=dec[0]
        # Decoder block 1: in=dec[0],       skip=s1, out=dec[1]
        # Decoder block 2: in=dec[1],       skip=s0, out=dec[2]
        # Decoder block 3: in=dec[2],       skip=—,  handled by final upsample

        self.blocks = nn.ModuleList()

        # Block 0: deepest → fuse with stage 2
        self.blocks.append(
            DecoderBlock(encoder_channels[3], encoder_channels[2], decoder_channels[0])
        )
        # Block 1: fuse with stage 1
        self.blocks.append(
            DecoderBlock(decoder_channels[0], encoder_channels[1], decoder_channels[1])
        )
        # Block 2: fuse with stage 0
        self.blocks.append(
            DecoderBlock(decoder_channels[1], encoder_channels[0], decoder_channels[2])
        )

        # Final upsample to input resolution (stage 0 is typically 1/4,
        # so we need one more 2x upsample + refine).
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        # Segmentation head — raw logits, no activation.
        self.seg_head = nn.Conv2d(decoder_channels[3], 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode encoder features into a segmentation logit map.

        Args:
            features: List of 4 feature tensors from the encoder, ordered
                      from shallowest (highest resolution) to deepest.

        Returns:
            Segmentation logits of shape ``[B, 1, H', W']`` where ``H', W'``
            is 2x the resolution of the shallowest encoder stage
            (typically half the input resolution).  Use bilinear
            interpolation outside if full-resolution output is needed.
        """
        s0, s1, s2, s3 = features

        x = self.blocks[0](s3, s2)   # deepest + skip from stage 2
        x = self.blocks[1](x, s1)    # + skip from stage 1
        x = self.blocks[2](x, s0)    # + skip from stage 0

        x = self.final_upsample(x)   # 2x upsample to ~input resolution
        logits = self.seg_head(x)     # [B, 1, H, W]
        return logits
