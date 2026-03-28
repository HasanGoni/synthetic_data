"""
Epic 5 — MC Dropout uncertainty estimation for segmentation models.

Enables Monte-Carlo Dropout at inference time to approximate Bayesian
uncertainty.  Multiple stochastic forward passes produce a distribution
of predictions; the entropy of that distribution quantifies how
uncertain the model is about each pixel / image.

Usage::

    from udm_epic5.uncertainty import mc_dropout_uncertainty

    df = mc_dropout_uncertainty(model, target_dataset, n_forward=20)
    print(df[["image_path", "mean_entropy", "prediction_variance"]])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Activate MC-Dropout mode on *model*.

    All ``Dropout`` (and ``Dropout2d`` / ``Dropout3d``) layers are set to
    ``train()`` so they keep sampling masks, while every other layer stays
    in ``eval()`` mode (BatchNorm statistics frozen, etc.).

    Args:
        model: A PyTorch module (modified **in-place**).

    Returns:
        The same model reference, for convenience chaining.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
    return model


def compute_entropy(prob_stack: Tensor) -> Tensor:
    """
    Per-pixel predictive entropy from *T* stochastic forward passes.

    Args:
        prob_stack: Probability maps ``[T, B, 1, H, W]`` where *T* is the
            number of forward passes.  Values should lie in ``[0, 1]``.

    Returns:
        Entropy map ``[B, 1, H, W]``.  Higher values indicate greater
        uncertainty.
    """
    eps = 1e-8
    # Mean probability across T forward passes -> [B, 1, H, W]
    mean_prob = prob_stack.mean(dim=0)

    # Binary entropy: -[ p*log(p) + (1-p)*log(1-p) ]
    entropy = -(
        mean_prob * torch.log(mean_prob + eps)
        + (1.0 - mean_prob) * torch.log(1.0 - mean_prob + eps)
    )
    return entropy


# ------------------------------------------------------------------
# Main estimation routine
# ------------------------------------------------------------------


def mc_dropout_uncertainty(
    model: nn.Module,
    dataset: Dataset,
    n_forward: int = 20,
    device: str = "cuda",
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Estimate per-image uncertainty via Monte-Carlo Dropout.

    The model is placed into MC-Dropout mode (dropout layers active,
    everything else in eval) and ``n_forward`` stochastic forward passes
    are executed for each sample.

    Args:
        model:      A segmentation model whose ``forward`` returns logits
                    ``[B, 1, H, W]`` (or a tuple whose first element is the
                    logits — compatible with :class:`DANNModel`).
        dataset:    A map-style dataset.  Each item must yield at least an
                    image tensor.  If the item is a ``dict`` it should
                    contain ``"image"`` and optionally ``"image_path"``.
                    If the item is a ``tuple`` the first element is the
                    image and the third (index 2) is treated as the path.
        n_forward:  Number of stochastic forward passes.
        device:     Target device (``"cuda"`` or ``"cpu"``).
        batch_size: DataLoader batch size.

    Returns:
        :class:`pandas.DataFrame` with columns:

        * ``image_path`` — file path (or ``"sample_{idx}"`` fallback)
        * ``mean_entropy`` — average per-pixel entropy
        * ``max_entropy`` — maximum per-pixel entropy
        * ``prediction_variance`` — mean per-pixel prediction variance
        * ``image_idx`` — integer index into *dataset*
    """
    model = enable_mc_dropout(model.to(device))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    records: list[dict[str, Any]] = []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            images, paths = _unpack_batch(batch, global_idx)
            images = images.to(device)
            current_bs = images.size(0)

            # Collect T stochastic forward passes -------------------------
            prob_list: list[Tensor] = []
            for _ in range(n_forward):
                logits = _forward_logits(model, images)
                probs = torch.sigmoid(logits)  # [B, 1, H, W]
                prob_list.append(probs)

            # Stack -> [T, B, 1, H, W]
            prob_stack = torch.stack(prob_list, dim=0)

            # Entropy map [B, 1, H, W]
            entropy_map = compute_entropy(prob_stack)

            # Prediction variance across T passes -> [T, B, 1, H, W]
            variance_map = prob_stack.var(dim=0)  # [B, 1, H, W]

            # Per-image statistics ----------------------------------------
            for i in range(current_bs):
                ent_i = entropy_map[i]  # [1, H, W]
                var_i = variance_map[i]

                records.append(
                    {
                        "image_path": paths[i],
                        "mean_entropy": ent_i.mean().item(),
                        "max_entropy": ent_i.max().item(),
                        "prediction_variance": var_i.mean().item(),
                        "image_idx": global_idx + i,
                    }
                )

            global_idx += current_bs

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    """Extract segmentation logits, handling both plain and DANN models."""
    out = model(images)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _unpack_batch(
    batch: Any,
    global_idx: int,
) -> tuple[Tensor, list[str]]:
    """
    Normalise a heterogeneous batch into ``(images, paths)``.

    Supports:
    * ``dict`` with ``"image"`` and optional ``"image_path"`` keys
    * ``tuple / list`` where ``[0]`` is the image tensor and ``[2]``
      (if present) is the path
    * a bare ``Tensor``
    """
    if isinstance(batch, dict):
        images = batch["image"]
        raw_paths = batch.get("image_path", None)
        if raw_paths is not None:
            paths = list(raw_paths) if not isinstance(raw_paths, str) else [raw_paths]
        else:
            paths = [f"sample_{global_idx + i}" for i in range(images.size(0))]
        return images, paths

    if isinstance(batch, (tuple, list)):
        images = batch[0]
        if len(batch) >= 3 and isinstance(batch[2], (list, tuple, str)):
            raw_paths = batch[2]
            paths = list(raw_paths) if not isinstance(raw_paths, str) else [raw_paths]
        else:
            paths = [f"sample_{global_idx + i}" for i in range(images.size(0))]
        return images, paths

    # Bare tensor
    images = batch
    paths = [f"sample_{global_idx + i}" for i in range(images.size(0))]
    return images, paths
