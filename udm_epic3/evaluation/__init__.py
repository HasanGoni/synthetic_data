"""Evaluation modules for Epic 3 CycleGAN cross-modality translation.

Provides image-quality metrics (SSIM, FID), defect-preservation scoring
(Dice coefficient), and a directory-level evaluation pipeline for
assessing CycleGAN translation quality.
"""

from .quality_metrics import (
    compute_fid,
    compute_ssim,
    compute_defect_dice,
    evaluate_translation,
)

__all__ = [
    "compute_fid",
    "compute_ssim",
    "compute_defect_dice",
    "evaluate_translation",
]
