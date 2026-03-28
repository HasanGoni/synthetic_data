"""
UDM Epic 8 -- Universal Model Support & Integration.

Unified cross-modality synthetic data generation and model training system
that combines all previous epics (1-7) into a single pipeline.

Install extras: ``pip install -e ".[epic8]"``
"""

from udm_epic8.registry.modality_registry import ModalityRegistry
from udm_epic8.pipeline.unified import UnifiedPipeline, UnifiedPipelineConfig
from udm_epic8.export.dataset_export import (
    export_to_coco,
    export_to_yolo,
    export_to_hf,
    merge_datasets,
)
from udm_epic8.evaluation.cross_modality import (
    cross_modality_report,
    compare_real_vs_synthetic,
)

__all__ = [
    "ModalityRegistry",
    "UnifiedPipeline",
    "UnifiedPipelineConfig",
    "export_to_coco",
    "export_to_yolo",
    "export_to_hf",
    "merge_datasets",
    "cross_modality_report",
    "compare_real_vs_synthetic",
]
