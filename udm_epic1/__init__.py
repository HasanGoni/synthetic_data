"""
UDM Epic 1 — Synthetic X-ray Void Dataset Generator
====================================================

Physics-based synthetic data generation for Universal Detection Model.
Uses Beer-Lambert law to simulate X-ray void contrast in solder joints.

Quick start
-----------
>>> from udm_epic1 import SyntheticSampleGenerator, GeneratorConfig
>>> gen = SyntheticSampleGenerator(GeneratorConfig(), seed=42)
>>> image, mask, meta = gen.generate(image_id="sample_0001")

CLI
---
    udm-generate run --config configs/default.yaml
    udm-generate preview --n 8
    udm-generate validate
    udm-generate stats
"""

__version__ = "0.1.0"
__author__ = "Mohammed Hasan Goni"

from udm_epic1.physics.beer_lambert import BeerLambertSimulator, BeerLambertConfig
from udm_epic1.generators.void_shapes import VoidShapeGenerator, VoidGeometry
from udm_epic1.generators.sample_generator import (
    SyntheticSampleGenerator,
    GeneratorConfig,
    SampleMeta,
)
from udm_epic1.augmentation.transforms import AugmentationPipeline, AugConfig
from udm_epic1.dataset.pipeline import DatasetPipeline, PipelineConfig

__all__ = [
    "BeerLambertSimulator",
    "BeerLambertConfig",
    "VoidShapeGenerator",
    "VoidGeometry",
    "SyntheticSampleGenerator",
    "GeneratorConfig",
    "SampleMeta",
    "AugmentationPipeline",
    "AugConfig",
    "DatasetPipeline",
    "PipelineConfig",
]
