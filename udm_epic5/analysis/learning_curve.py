"""
Epic 5 — Learning-curve tracking and visualisation.

Collects per-round evaluation metrics (IoU, Dice, etc.) and generates
comparison plots across active-learning strategies.

Usage::

    from udm_epic5.analysis.learning_curve import (
        build_learning_curve_df, check_stopping_criterion, plot_learning_curves,
    )

    df = build_learning_curve_df("outputs/epic5_active/results.csv")
    stop = check_stopping_criterion(df, metric="dice", threshold=0.02)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------
# DataFrame construction
# ------------------------------------------------------------------

REQUIRED_COLUMNS = ["round", "strategy", "budget", "dice", "iou", "f1"]


def build_learning_curve_df(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a results CSV and validate expected columns.

    The CSV must contain at least: ``round``, ``strategy``, ``budget``,
    ``dice``, ``iou``, ``f1``.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Validated :class:`~pandas.DataFrame`.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")
    return df


# ------------------------------------------------------------------
# Stopping criterion
# ------------------------------------------------------------------


def check_stopping_criterion(
    df: pd.DataFrame,
    metric: str = "dice",
    threshold: float = 0.02,
) -> bool:
    """
    Return ``True`` if the improvement from the last two rounds is below
    *threshold*, indicating diminishing returns.

    Args:
        df:        Learning-curve dataframe (must contain *metric* and
                   ``round`` columns).
        metric:    Column name for the evaluation metric.
        threshold: Minimum improvement to continue.

    Returns:
        ``True`` if training should stop (improvement < *threshold*).
    """
    if len(df) < 2:
        return False

    sorted_df = df.sort_values("round")
    last_two = sorted_df[metric].values[-2:]
    improvement = last_two[1] - last_two[0]
    return float(improvement) < threshold


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def plot_learning_curves(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("dice", "iou"),
    save_dir: str | Path = ".",
) -> list[Path]:
    """
    Generate per-metric learning-curve plots grouped by strategy.

    Args:
        df:       Learning-curve dataframe.
        metrics:  Which columns to plot.
        save_dir: Output directory for PNG files.

    Returns:
        List of saved file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, group in df.groupby("strategy"):
            group_sorted = group.sort_values("round")
            ax.plot(
                group_sorted["budget"].cumsum(),
                group_sorted[metric],
                marker="o",
                label=str(strategy),
            )
        ax.set_xlabel("Cumulative labeled samples")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Active Learning — {metric.upper()} vs. Labeled Budget")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = save_dir / f"learning_curve_{metric}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
