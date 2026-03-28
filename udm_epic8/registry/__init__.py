"""Modality registry for Epic 8 universal pipeline.

Provides a central :class:`ModalityRegistry` that maps modality names to
their generator functions and configuration classes.  All previous epics
are pre-registered at import time.
"""

from .modality_registry import ModalityRegistry, registry

__all__ = [
    "ModalityRegistry",
    "registry",
]
