"""Data loading modules for Epic 3 CycleGAN cross-modality translation.

Provides unpaired and paired dataset handling for CycleGAN training
between AOI and USM inspection modalities.
"""

from .unpaired_dataset import UnpairedDataset, PairedDataset, build_cyclegan_datasets

__all__ = [
    "UnpairedDataset",
    "PairedDataset",
    "build_cyclegan_datasets",
]
