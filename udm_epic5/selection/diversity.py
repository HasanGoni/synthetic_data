"""
Epic 5 — Diversity-based sample selection strategies.

Provides two complementary approaches for picking a representative subset
from an unlabelled target-domain pool:

* **Coreset (farthest-first traversal)** — greedily maximises the minimum
  distance between selected samples in feature space.
* **Clustering (k-means)** — partitions the pool into *k* clusters and
  picks the sample closest to each centroid.

Both operate on pre-extracted feature vectors (e.g. from the DANN
bottleneck) and return integer indices into the original feature array.

Usage::

    from udm_epic5.selection.diversity import coreset_selection

    indices = coreset_selection(features, budget=50)
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# Coreset (farthest-first traversal)
# ------------------------------------------------------------------


def coreset_selection(
    features: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Greedy farthest-first traversal (k-Center-Greedy).

    Starting from a random seed point, iteratively select the sample
    whose minimum distance to any already-selected sample is largest.
    This maximises geometric coverage of the feature space.

    Args:
        features: Feature matrix ``[N, feat_dim]``.
        budget:   Number of samples to select.
        seed:     Random seed for the initial point.

    Returns:
        Integer index array ``[budget]`` into *features*.
    """
    n_samples = features.shape[0]
    budget = min(budget, n_samples)

    rng = np.random.RandomState(seed)
    selected: list[int] = [rng.randint(0, n_samples)]

    # min_distances[i] = distance from sample i to the nearest selected sample
    min_distances = np.full(n_samples, np.inf, dtype=np.float64)

    for _ in range(budget - 1):
        last = selected[-1]
        # Distance from every sample to the most recently added point
        dists = np.linalg.norm(
            features - features[last][np.newaxis, :], axis=1
        )
        min_distances = np.minimum(min_distances, dists)

        # Exclude already-selected points
        min_distances_copy = min_distances.copy()
        min_distances_copy[selected] = -1.0

        next_idx = int(np.argmax(min_distances_copy))
        selected.append(next_idx)

    return np.array(selected, dtype=np.intp)


# ------------------------------------------------------------------
# Clustering (k-means)
# ------------------------------------------------------------------


def clustering_selection(
    features: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """
    k-Means clustering followed by nearest-to-centroid selection.

    Fits *k* = ``budget`` clusters on *features* and, for each cluster,
    returns the index of the sample closest to its centroid.

    Args:
        features: Feature matrix ``[N, feat_dim]``.
        budget:   Number of samples (clusters) to select.
        seed:     Random seed for k-means initialisation.

    Returns:
        Integer index array ``[budget]`` into *features*.
    """
    n_samples = features.shape[0]
    budget = min(budget, n_samples)

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    kmeans = KMeans(n_clusters=budget, random_state=seed, n_init=10)
    kmeans.fit(features)

    centroids = kmeans.cluster_centers_  # [budget, feat_dim]
    # Distance from every centroid to every sample -> [budget, N]
    dists = pairwise_distances(centroids, features)

    selected: list[int] = []
    used: set[int] = set()

    for k in range(budget):
        # Indices sorted by distance to centroid k
        order = np.argsort(dists[k])
        for idx in order:
            idx = int(idx)
            if idx not in used:
                selected.append(idx)
                used.add(idx)
                break

    return np.array(selected, dtype=np.intp)
