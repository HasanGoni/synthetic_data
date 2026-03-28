"""Translation (inference) modules for Epic 3 CycleGAN.

Provides functions to translate single images or entire directories
between modalities using a trained CycleGAN checkpoint.
"""

from .translate import translate_directory, translate_single

__all__ = [
    "translate_directory",
    "translate_single",
]
