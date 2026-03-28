"""Pipeline modules for Epic 8 universal model support.

Provides the unified pipeline configuration and orchestrator that dispatches
synthetic data generation across all supported modalities (xray, aoi, usm,
chromasense, etc.).
"""

from .unified import UnifiedPipeline, UnifiedPipelineConfig

__all__ = [
    "UnifiedPipeline",
    "UnifiedPipelineConfig",
]
