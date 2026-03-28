"""Training modules for Epic 3 CycleGAN cross-modality translation.

Provides the full CycleGAN adversarial training loop with LSGAN loss,
cycle-consistency, identity regularisation, and optional defect-preservation
loss for AOI/USM translation.
"""

from .train_cyclegan import train_cyclegan_from_yaml

__all__ = [
    "train_cyclegan_from_yaml",
]
