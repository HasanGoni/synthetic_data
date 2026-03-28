"""Training modules for Epic 4 DANN (Domain-Adaptive Neural Network).

Provides source-only baseline training, DANN adversarial training, and the
GRL lambda schedule used to ramp domain-adaptation strength over epochs.
"""

from .lambda_scheduler import dann_lambda_schedule
from .train_baseline import bce_dice_loss, train_baseline_from_yaml
from .train_dann import train_dann_from_yaml

__all__ = [
    "dann_lambda_schedule",
    "bce_dice_loss",
    "train_baseline_from_yaml",
    "train_dann_from_yaml",
]
