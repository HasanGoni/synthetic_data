"""Evaluation modules for Epic 8 universal pipeline.

Provides cross-modality reporting and real-vs-synthetic quality comparison
utilities for assessing the unified dataset.
"""

from .cross_modality import (
    compare_real_vs_synthetic,
    cross_modality_report,
)

__all__ = [
    "cross_modality_report",
    "compare_real_vs_synthetic",
]
