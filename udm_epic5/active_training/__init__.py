"""Active training modules for Epic 5 — Active Domain Adaptation.

Extends the Epic 4 DANN training loop with support for labeled target
samples selected through active learning strategies.
"""

from .train_active_dann import train_active_dann_from_yaml

__all__ = [
    "train_active_dann_from_yaml",
]
