"""
Epic 4 — Gradient Reversal Layer and Domain Classifier.

Implements the domain-adversarial head described in *Ganin et al., 2016*
("Domain-Adversarial Training of Neural Networks").  During the forward
pass the features pass through unchanged; during the backward pass the
gradients are *negated* and scaled by ``lambda_val``, encouraging the
shared encoder to produce domain-invariant representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


# ── Gradient Reversal ────────────────────────────────────────────────────────


class GradientReversalFunction(Function):
    """Autograd function that reverses gradients during backprop."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Thin ``nn.Module`` wrapper around :class:`GradientReversalFunction`.

    Args:
        lambda_val: Scaling factor applied to the reversed gradient.
                    Typically annealed from 0 to 1 over training.
    """

    def __init__(self, lambda_val: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)


# ── Domain Classifier Head ───────────────────────────────────────────────────


class DomainClassifier(nn.Module):
    """
    Binary domain classifier applied to encoder bottleneck features.

    Architecture::

        AdaptiveAvgPool2d(1) -> Flatten
        -> FC(feature_dim, hidden_dim) -> ReLU -> Dropout(0.5)
        -> FC(hidden_dim, 1)

    Returns raw logits (apply ``torch.sigmoid`` or use
    ``BCEWithLogitsLoss`` downstream).

    Args:
        feature_dim: Channel count of the input feature map (the encoder's
                     deepest stage).
        hidden_dim:  Width of the hidden fully-connected layer.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, lambda_val: float = 1.0,
    ) -> torch.Tensor:
        """
        Classify the domain of the input features.

        Args:
            features:   Bottleneck feature map ``[B, C, H, W]``.
            lambda_val: GRL scaling factor (gradient reversal strength).

        Returns:
            Domain logits of shape ``[B, 1]``.
        """
        self.grl.lambda_val = lambda_val
        x = self.grl(features)
        x = self.pool(x)
        return self.classifier(x)
