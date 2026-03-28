"""
UDM Epic 3 — CycleGAN Cross-Modality Translation.

Train CycleGAN models for unpaired image-to-image translation between
AOI and USM inspection modalities, with optional defect-preservation
loss for maintaining defect fidelity during translation.

Install extras: ``pip install -e ".[epic3]"``
"""

from udm_epic3.models.cyclegan import CycleGANModel
from udm_epic3.models.generator import ResnetGenerator
from udm_epic3.models.discriminator import PatchDiscriminator
from udm_epic3.evaluation.quality_metrics import (
    compute_fid,
    compute_ssim,
    compute_defect_dice,
    evaluate_translation,
)

__all__ = [
    "CycleGANModel",
    "ResnetGenerator",
    "PatchDiscriminator",
    "compute_fid",
    "compute_ssim",
    "compute_defect_dice",
    "evaluate_translation",
]
