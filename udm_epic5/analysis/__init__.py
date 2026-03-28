"""Analysis modules for Epic 5 — multi-round convergence and strategy comparison.

Provides tools to build learning curves, compare active learning strategies,
and determine when to stop the active learning loop.
"""

from .convergence import compare_strategies, learning_curve, stopping_criterion

__all__ = [
    "learning_curve",
    "compare_strategies",
    "stopping_criterion",
]
