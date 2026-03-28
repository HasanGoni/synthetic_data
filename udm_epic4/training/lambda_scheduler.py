"""
Epic 4 — Gradient Reversal Layer lambda scheduler.

Implements the progressive lambda schedule from Ganin & Lempitsky (2015):
the domain loss weight ramps from 0 to *lambda_max* following a sigmoid
curve, preventing early-training instability when the encoder has not yet
learned useful features.
"""

from __future__ import annotations

import math


def dann_lambda_schedule(
    progress: float,
    lambda_max: float = 1.0,
) -> float:
    """
    Compute the GRL lambda for the current training progress.

    Uses the schedule from *Domain-Adversarial Training of Neural Networks*
    (Ganin et al., 2016):

        lambda = lambda_max * (2 / (1 + exp(-10 * progress)) - 1)

    At ``progress = 0`` the value is ~0 (domain head disabled); at
    ``progress = 1`` it saturates near *lambda_max*.

    Args:
        progress:   Fraction of training completed, clamped to [0, 1].
                    Typically ``current_epoch / max_epochs``.
        lambda_max: Ceiling value for lambda (default ``1.0``).

    Returns:
        Current lambda value as a float.
    """
    progress = max(0.0, min(1.0, progress))
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
