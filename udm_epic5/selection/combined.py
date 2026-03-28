"""
Epic 5 — Combined uncertainty + diversity sample selection.

Merges an uncertainty signal (e.g. MC-Dropout entropy) with a
diversity objective so the selected annotation budget covers both
*informative* and *representative* samples from the unlabelled
target-domain pool.

Usage::

    from udm_epic5.selection.combined import combined_selection

    indices = combined_selection(
        uncertainty_scores=df["mean_entropy"].values,
        features=bottleneck_features,
        budget=50,
        alpha=0.7,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Combined selection
# ------------------------------------------------------------------


def combined_selection(
    uncertainty_scores: np.ndarray,
    features: np.ndarray,
    budget: int,
    alpha: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """
    Greedy iterative selection balancing uncertainty and diversity.

    At each step the next sample is chosen to maximise:

        ``score = alpha * norm_uncertainty + (1 - alpha) * norm_diversity``

    where *norm_uncertainty* is the min-max normalised uncertainty and
    *norm_diversity* is the (normalised) distance from the candidate to
    its nearest already-selected sample.

    Args:
        uncertainty_scores: Per-sample uncertainty ``[N]`` (higher = more
            uncertain).
        features:           Feature matrix ``[N, feat_dim]`` for distance
            computation.
        budget:             Number of samples to select.
        alpha:              Trade-off weight (``1.0`` = pure uncertainty,
            ``0.0`` = pure diversity).
        seed:               Random seed for tie-breaking / first selection.

    Returns:
        Integer index array ``[budget]`` of selected samples.
    """
    n_samples = features.shape[0]
    budget = min(budget, n_samples)

    rng = np.random.RandomState(seed)

    # Normalise uncertainty to [0, 1] ---------------------------------
    u = uncertainty_scores.astype(np.float64).copy()
    u_min, u_max = u.min(), u.max()
    if u_max - u_min > 0:
        u_norm = (u - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(u)

    # Track minimum distance to any selected sample -------------------
    min_dist = np.full(n_samples, np.inf, dtype=np.float64)

    selected: list[int] = []
    mask = np.ones(n_samples, dtype=bool)  # True = still available

    # First selection: highest uncertainty (break ties randomly) -------
    top_u = np.where(u_norm == u_norm.max())[0]
    first = int(rng.choice(top_u))
    selected.append(first)
    mask[first] = False

    # Update distances after first pick
    dists = np.linalg.norm(features - features[first][np.newaxis, :], axis=1)
    min_dist = np.minimum(min_dist, dists)

    # Greedy loop -----------------------------------------------------
    for _ in range(budget - 1):
        # Normalise diversity scores (min-dist) among remaining samples
        d = min_dist.copy()
        d[~mask] = -np.inf  # exclude already picked

        d_valid = d[mask]
        d_min, d_max = d_valid.min(), d_valid.max()
        if d_max - d_min > 0:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)
        d_norm[~mask] = -np.inf

        # Combined score
        combined = alpha * u_norm + (1.0 - alpha) * d_norm
        combined[~mask] = -np.inf

        next_idx = int(np.argmax(combined))
        selected.append(next_idx)
        mask[next_idx] = False

        # Update min distances
        dists = np.linalg.norm(
            features - features[next_idx][np.newaxis, :], axis=1
        )
        min_dist = np.minimum(min_dist, dists)

    return np.array(selected, dtype=np.intp)


# ------------------------------------------------------------------
# Export helper
# ------------------------------------------------------------------


def export_selection_csv(
    selected_indices: np.ndarray,
    uncertainty_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Write a CSV summarising the selected samples for labelling.

    The output is designed for direct ingestion by annotation tooling
    (e.g. CVAT, Label Studio).

    Args:
        selected_indices: Ordered array of selected row indices
            (position = rank).
        uncertainty_df:   DataFrame produced by
            :func:`mc_dropout_uncertainty`, must contain at least
            ``image_path``, ``mean_entropy``, and ``max_entropy`` columns.
        output_path:      Destination CSV path (parent dirs created
            automatically).

    Produces columns:

    * ``rank`` — selection order (1-based)
    * ``image_path``
    * ``mean_entropy``
    * ``max_entropy``
    * ``selected_reason`` — human-readable tag
    """
    rows: list[dict[str, object]] = []

    for rank_0, idx in enumerate(selected_indices):
        row = uncertainty_df.iloc[idx]
        mean_ent = float(row["mean_entropy"])
        max_ent = float(row["max_entropy"])

        # Determine a human-readable reason
        reason = _classify_reason(mean_ent, uncertainty_df["mean_entropy"])

        rows.append(
            {
                "rank": rank_0 + 1,
                "image_path": row["image_path"],
                "mean_entropy": mean_ent,
                "max_entropy": max_ent,
                "selected_reason": reason,
            }
        )

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _classify_reason(
    mean_entropy: float,
    all_entropies: pd.Series,
) -> str:
    """Return a short tag describing why a sample was selected."""
    q75 = float(all_entropies.quantile(0.75))
    q25 = float(all_entropies.quantile(0.25))

    if mean_entropy >= q75:
        return "high_uncertainty"
    if mean_entropy <= q25:
        return "diversity"
    return "uncertainty_and_diversity"
