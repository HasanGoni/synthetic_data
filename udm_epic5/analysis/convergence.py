"""
Epic 5 — Multi-round convergence analysis for active domain adaptation.

Provides utilities to:
  - Aggregate per-round results into a learning curve DataFrame.
  - Plot F1 vs. number of labels for each selection strategy.
  - Decide whether additional labeling rounds are worthwhile.

Typical usage::

    from udm_epic5.analysis.convergence import (
        learning_curve,
        compare_strategies,
        stopping_criterion,
    )

    results = [
        {"n_labels": 10, "strategy": "uncertainty", "f1": 0.65, "iou": 0.50, "round": 1},
        {"n_labels": 20, "strategy": "uncertainty", "f1": 0.72, "iou": 0.58, "round": 2},
        ...
    ]
    df = learning_curve(results)
    compare_strategies(df, output_path="convergence.png")
    should_stop = stopping_criterion(results, min_improvement=0.02)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Use non-interactive backend when saving plots headlessly.
matplotlib.use("Agg")

# Canonical strategy colours for consistent plotting.
_STRATEGY_COLOURS = {
    "random": "#7f7f7f",
    "uncertainty": "#1f77b4",
    "diversity": "#2ca02c",
    "combined": "#d62728",
    "active_dann": "#ff7f0e",
}

# Marker styles per strategy.
_STRATEGY_MARKERS = {
    "random": "s",
    "uncertainty": "o",
    "diversity": "^",
    "combined": "D",
    "active_dann": "*",
}


# ------------------------------------------------------------------
# Learning curve construction
# ------------------------------------------------------------------


def learning_curve(results: List[dict]) -> pd.DataFrame:
    """Convert a list of per-round result dicts into a tidy DataFrame.

    Each dict is expected to contain at minimum:

    - ``n_labels`` (int): cumulative number of labeled target samples.
    - ``strategy`` (str): selection strategy name.
    - ``f1`` (float): segmentation F1 score on the target validation set.
    - ``iou`` (float): segmentation IoU on the target validation set.
    - ``round`` (int): active learning round number.

    Additional keys (e.g. ``dice``, ``precision``, ``recall``) are preserved
    as extra columns.

    Args:
        results: List of result dictionaries, one per round per strategy.

    Returns:
        :class:`pandas.DataFrame` sorted by ``(strategy, round)``.

    Raises:
        ValueError: If *results* is empty or missing required keys.
    """
    if not results:
        raise ValueError("results list must not be empty")

    required_keys = {"n_labels", "strategy", "f1", "iou", "round"}
    for i, r in enumerate(results):
        missing = required_keys - set(r.keys())
        if missing:
            raise ValueError(
                f"Result dict at index {i} is missing keys: {missing}"
            )

    df = pd.DataFrame(results)
    df = df.sort_values(["strategy", "round"]).reset_index(drop=True)

    logger.info(
        "Built learning curve: %d rows, strategies=%s",
        len(df),
        sorted(df["strategy"].unique()),
    )
    return df


# ------------------------------------------------------------------
# Strategy comparison plot
# ------------------------------------------------------------------


def compare_strategies(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Plot F1 vs. number of labels for each selection strategy.

    Produces a line plot with one curve per strategy, using consistent
    colours and markers.  If *output_path* is provided the figure is saved
    to disk; otherwise it is displayed via ``plt.show()``.

    Args:
        results_df: DataFrame produced by :func:`learning_curve`, with at
                    least columns ``n_labels``, ``strategy``, ``f1``.
        output_path: Optional path to save the figure (e.g. ``"plot.png"``).
                     Supports any format accepted by matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = sorted(results_df["strategy"].unique())
    for strategy in strategies:
        subset = results_df[results_df["strategy"] == strategy].sort_values("n_labels")
        colour = _STRATEGY_COLOURS.get(strategy, None)
        marker = _STRATEGY_MARKERS.get(strategy, "o")

        ax.plot(
            subset["n_labels"],
            subset["f1"],
            label=strategy,
            color=colour,
            marker=marker,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Number of Labeled Target Samples", fontsize=12)
    ax.set_ylabel("Target F1 Score", fontsize=12)
    ax.set_title("Active Learning: F1 vs. Label Budget", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0, top=1.05)

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Saved strategy comparison plot to %s", out)
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Stopping criterion
# ------------------------------------------------------------------


def stopping_criterion(
    results: List[dict],
    min_improvement: float = 0.02,
) -> bool:
    """Determine whether to stop the active learning loop.

    Compares the F1 of the most recent round to the previous round.  If
    the improvement is below *min_improvement*, returns ``True`` to signal
    that further labeling is unlikely to yield significant gains.

    When results contain multiple strategies, the criterion is evaluated
    on the strategy present in the last result entry.

    Args:
        results: List of result dicts (same format as :func:`learning_curve`).
                 Must have at least two entries for the relevant strategy.
        min_improvement: Minimum F1 improvement threshold.  If the last
                         round improved by less than this, we recommend
                         stopping.  Default ``0.02``.

    Returns:
        ``True`` if the last round improved F1 by less than *min_improvement*
        (i.e. we should stop); ``False`` if improvement is still significant.

    Raises:
        ValueError: If fewer than 2 results are available for the strategy.
    """
    if len(results) < 2:
        raise ValueError(
            "Need at least 2 result entries to evaluate stopping criterion"
        )

    # Focus on the strategy of the last result
    last_strategy = results[-1]["strategy"]
    strategy_results = [r for r in results if r["strategy"] == last_strategy]
    strategy_results.sort(key=lambda r: r["round"])

    if len(strategy_results) < 2:
        raise ValueError(
            f"Need at least 2 rounds for strategy '{last_strategy}', "
            f"found {len(strategy_results)}"
        )

    prev_f1 = strategy_results[-2]["f1"]
    curr_f1 = strategy_results[-1]["f1"]
    improvement = curr_f1 - prev_f1

    logger.info(
        "Stopping criterion: strategy=%s, prev_f1=%.4f, curr_f1=%.4f, "
        "improvement=%.4f, threshold=%.4f, should_stop=%s",
        last_strategy,
        prev_f1,
        curr_f1,
        improvement,
        min_improvement,
        improvement < min_improvement,
    )

    return improvement < min_improvement
