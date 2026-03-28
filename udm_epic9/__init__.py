"""
UDM Epic 9 — Synthetic Crack Generation for semiconductor defect detection.

Provides three core capabilities:

1. **Synthetic crack generation** — overlay realistic crack patterns onto USM images
2. **Domain transfer** — translate USM crack images to RGB domain appearance
3. **Mask-to-image generation** — given only a crack mask, generate a domain image

Install extras: ``pip install -e ".[epic9]"``
"""

from __future__ import annotations

from udm_epic9.models.crack_geometry import (
    CrackProfile,
    generate_crack_path,
    generate_branching_crack,
    generate_crack_network,
    render_crack_mask,
)
from udm_epic9.models.crack_types import (
    die_crack,
    substrate_crack,
    mold_crack,
    delamination_crack,
)
from udm_epic9.rendering.usm_renderer import (
    render_crack_on_usm,
    generate_synthetic_usm_with_cracks,
)
from udm_epic9.domain_transfer.usm_to_rgb import (
    USMtoRGBTransfer,
    mask_to_image,
)

__all__ = [
    "CrackProfile",
    "generate_crack_path",
    "generate_branching_crack",
    "generate_crack_network",
    "render_crack_mask",
    "die_crack",
    "substrate_crack",
    "mold_crack",
    "delamination_crack",
    "render_crack_on_usm",
    "generate_synthetic_usm_with_cracks",
    "USMtoRGBTransfer",
    "mask_to_image",
]
