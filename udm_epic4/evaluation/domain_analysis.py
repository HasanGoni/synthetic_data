"""
Epic 4 — Domain analysis utilities for DANN evaluation.

Provides feature extraction, t-SNE visualisation, and a domain confusion
score that quantifies how well the encoder has learned domain-invariant
representations.  A lower confusion score (closer to 0.5 accuracy)
indicates stronger domain adaptation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Feature extraction ──────────────────────────────────────────────────────


def _collate_for_features(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Minimal collate: stack images and collect domain labels."""
    images = torch.stack([s["image"] for s in batch])
    domains = torch.tensor([s["domain"] for s in batch], dtype=torch.long)
    return {"image": images, "domain": domains}


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: Any,
    device: str = "cuda",
    max_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract bottleneck features from *model* for each sample in *dataset*.

    The model is expected to expose an ``encode(x)`` method that returns
    a list of multi-scale feature tensors (as produced by
    :class:`SharedEncoder`).  The *deepest* feature map (last element) is
    global-average-pooled to yield a 1-D feature vector per sample.

    If the model does not have ``encode``, the function falls back to
    calling ``model.encoder(x)`` or, as a last resort, the full forward
    pass and treats the output as the feature.

    Args:
        model:       Trained DANN model (or its encoder component).
        dataset:     Dataset following the :class:`DomainDataset` interface.
        device:      Torch device string.
        max_samples: Cap on the number of samples to process (controls
                     runtime for large datasets).

    Returns:
        Tuple of:
            - **features** — ``np.ndarray`` of shape ``[N, feat_dim]``.
            - **domain_labels** — ``np.ndarray`` of shape ``[N]`` (int).
    """
    model.eval()

    # Determine how many samples to use.
    n_samples = min(len(dataset), max_samples)
    indices = list(range(n_samples))
    subset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_for_features,
    )

    all_features: list[np.ndarray] = []
    all_domains: list[np.ndarray] = []
    pool = nn.AdaptiveAvgPool2d(1)

    for batch in loader:
        images = batch["image"].to(device)
        domains = batch["domain"].numpy()

        # Obtain the bottleneck feature map.
        feat_map = _get_bottleneck(model, images)

        # Global average pool: [B, C, H, W] -> [B, C]
        pooled = pool(feat_map).flatten(start_dim=1)
        all_features.append(pooled.cpu().numpy())
        all_domains.append(domains)

    features = np.concatenate(all_features, axis=0)
    domain_labels = np.concatenate(all_domains, axis=0)
    logger.info(
        "Extracted features: shape=%s  domains=%s",
        features.shape,
        np.unique(domain_labels).tolist(),
    )
    return features, domain_labels


def _get_bottleneck(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract the deepest spatial feature map from *model*.

    Tries, in order:
        1. ``model.encode(x)`` — expected to return a list of feature tensors.
        2. ``model.encoder(x)`` — :class:`SharedEncoder` forward.
        3. ``model(x)`` — treat output as the feature directly.
    """
    if hasattr(model, "encode"):
        features = model.encode(x)
        if isinstance(features, (list, tuple)):
            return features[-1]
        return features

    if hasattr(model, "encoder"):
        encoder = model.encoder
        features = encoder(x)
        if isinstance(features, (list, tuple)):
            return features[-1]
        return features

    # Fallback: full forward pass.
    out = model(x)
    if isinstance(out, (list, tuple)):
        return out[-1]
    return out


# ── t-SNE ───────────────────────────────────────────────────────────────────


def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Compute 2-D t-SNE embedding of *features*.

    Args:
        features:   Array of shape ``[N, feat_dim]``.
        perplexity: t-SNE perplexity (will be clamped to ``N - 1`` if the
                    dataset is very small).
        seed:       Random seed for reproducibility.

    Returns:
        ``np.ndarray`` of shape ``[N, 2]`` with the 2-D coordinates.
    """
    from sklearn.manifold import TSNE

    effective_perplexity = min(perplexity, max(1.0, features.shape[0] - 1.0))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(features)
    logger.info("t-SNE complete: output shape %s", coords.shape)
    return coords


def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    domain_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Create a scatter plot of t-SNE coordinates, coloured by domain.

    Args:
        coords:       ``[N, 2]`` array of t-SNE coordinates.
        labels:       ``[N]`` integer domain labels.
        domain_names: Human-readable name for each domain index.
        save_path:    If provided, the figure is saved to this path (PNG).
                      Otherwise the figure is closed without saving.
    """
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = domain_names[int(lbl)] if int(lbl) < len(domain_names) else f"domain_{lbl}"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(idx)],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title("t-SNE of Encoder Bottleneck Features")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=9, markerscale=2.0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("t-SNE plot saved to %s", save_path)
    plt.close(fig)


# ── Domain confusion score ──────────────────────────────────────────────────


@torch.no_grad()
def domain_confusion_score(
    model: nn.Module,
    source_dataset: Any,
    target_dataset: Any,
    device: str = "cuda",
    max_samples: int = 500,
) -> float:
    """Measure domain separability via a linear probe on frozen features.

    A small logistic-regression probe is trained on bottleneck features to
    classify source vs. target domain.  The reported accuracy is evaluated
    on a held-out split:

    - **Accuracy ~ 0.5** means the encoder has learned domain-invariant
      features (ideal for DANN).
    - **Accuracy ~ 1.0** means domains are still easily separable.

    Args:
        model:          Trained DANN model (or encoder).
        source_dataset: Source-domain dataset.
        target_dataset: Target-domain dataset.
        device:         Torch device string.
        max_samples:    Per-domain sample cap.

    Returns:
        Probe accuracy as a float in [0, 1].  Lower is better for
        domain adaptation.
    """
    # Extract features from both domains.
    src_feats, _ = extract_features(model, source_dataset, device=device, max_samples=max_samples)
    tgt_feats, _ = extract_features(model, target_dataset, device=device, max_samples=max_samples)

    n_src = src_feats.shape[0]
    n_tgt = tgt_feats.shape[0]

    features = np.concatenate([src_feats, tgt_feats], axis=0)
    labels = np.concatenate([np.zeros(n_src), np.ones(n_tgt)], axis=0)

    # Shuffle and split 70/30.
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(labels))
    features = features[perm]
    labels = labels[perm]

    split = int(0.7 * len(labels))
    train_x, val_x = features[:split], features[split:]
    train_y, val_y = labels[:split], labels[split:]

    # Train a small linear probe with SGD.
    feat_dim = train_x.shape[1]
    probe = nn.Linear(feat_dim, 1).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Train for a few epochs — we only need a rough separability estimate.
    probe.train()
    for _epoch in range(10):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            loss = criterion(probe(x_batch), y_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    # Evaluate on validation split.
    probe.eval()
    with torch.no_grad():
        val_logits = probe(torch.from_numpy(val_x).float().to(device))
        val_preds = (val_logits.squeeze(1) > 0.0).cpu().numpy()
        accuracy = float(np.mean(val_preds == val_y))

    logger.info(
        "Domain confusion probe accuracy: %.4f  (n_src=%d, n_tgt=%d)",
        accuracy,
        n_src,
        n_tgt,
    )
    return accuracy
