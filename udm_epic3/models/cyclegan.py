"""
Epic 3 — Full CycleGAN model for cross-modality translation.

Composes two ResNet generators (AOI -> USM and USM -> AOI) and two PatchGAN
discriminators into a single module that supports both training (forward
through both cycles) and single-direction inference.

Usage::

    from udm_epic3.models.cyclegan import CycleGANModel

    model = CycleGANModel(in_channels=1)
    outputs = model.forward_generators(real_A, real_B)
    fake_B = model.translate(real_A, direction="a2b")
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from udm_epic3.models.generator import ResnetGenerator
from udm_epic3.models.discriminator import PatchDiscriminator


class CycleGANModel(nn.Module):
    """
    Complete CycleGAN with two generator-discriminator pairs.

    Generators:
        * **G_A2B** — translates domain A (AOI) to domain B (USM).
        * **G_B2A** — translates domain B (USM) to domain A (AOI).

    Discriminators:
        * **D_A** — distinguishes real AOI images from fakes produced by G_B2A.
        * **D_B** — distinguishes real USM images from fakes produced by G_A2B.

    Args:
        in_channels:  Number of image channels (1 for grayscale).
        n_filters_g:  Base filter count for the generators.
        n_blocks:     Number of ResNet blocks in each generator.
        n_filters_d:  Base filter count for the discriminators.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_filters_g: int = 64,
        n_blocks: int = 9,
        n_filters_d: int = 64,
    ) -> None:
        super().__init__()

        # Generators
        self.G_A2B = ResnetGenerator(
            in_channels=in_channels,
            out_channels=in_channels,
            n_filters=n_filters_g,
            n_blocks=n_blocks,
        )
        self.G_B2A = ResnetGenerator(
            in_channels=in_channels,
            out_channels=in_channels,
            n_filters=n_filters_g,
            n_blocks=n_blocks,
        )

        # Discriminators
        self.D_A = PatchDiscriminator(
            in_channels=in_channels,
            n_filters=n_filters_d,
        )
        self.D_B = PatchDiscriminator(
            in_channels=in_channels,
            n_filters=n_filters_d,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_generators(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run both generators through the full CycleGAN forward pass.

        Computes fake images, cycle-reconstructed images, and identity
        mappings for both translation directions.

        Args:
            real_A: Batch of real domain-A (AOI) images ``[B, C, H, W]``.
            real_B: Batch of real domain-B (USM) images ``[B, C, H, W]``.

        Returns:
            Dictionary containing:
                * ``fake_B``  — G_A2B(real_A)
                * ``rec_A``   — G_B2A(fake_B)  (cycle A -> B -> A)
                * ``fake_A``  — G_B2A(real_B)
                * ``rec_B``   — G_A2B(fake_A)  (cycle B -> A -> B)
                * ``idt_A``   — G_B2A(real_A)  (identity mapping)
                * ``idt_B``   — G_A2B(real_B)  (identity mapping)
        """
        # Forward cycle: A -> B -> A
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)

        # Backward cycle: B -> A -> B
        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)

        # Identity mappings (generator should be identity on target domain)
        idt_A = self.G_B2A(real_A)
        idt_B = self.G_A2B(real_B)

        return {
            "fake_B": fake_B,
            "rec_A": rec_A,
            "fake_A": fake_A,
            "rec_B": rec_B,
            "idt_A": idt_A,
            "idt_B": idt_B,
        }

    @torch.no_grad()
    def translate(
        self,
        image: torch.Tensor,
        direction: str = "a2b",
    ) -> torch.Tensor:
        """
        Translate a single image for inference.

        Args:
            image:     Input image tensor ``[B, C, H, W]``.
            direction: Translation direction — ``"a2b"`` uses G_A2B
                       (AOI -> USM), ``"b2a"`` uses G_B2A (USM -> AOI).

        Returns:
            Translated image tensor ``[B, C, H, W]`` in ``[-1, 1]``.

        Raises:
            ValueError: If *direction* is not ``"a2b"`` or ``"b2a"``.
        """
        if direction == "a2b":
            return self.G_A2B(image)
        elif direction == "b2a":
            return self.G_B2A(image)
        else:
            raise ValueError(
                f"Invalid direction '{direction}'. Expected 'a2b' or 'b2a'."
            )
