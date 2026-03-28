#!/usr/bin/env python3
"""
Compare synthetic manifest.csv with optional real-mask statistics JSON.

Uses two-sample KS tests (scipy) on void count and area fraction.

Example:
    python scripts/validation/compare_distributions.py \\
        --manifest data/synthetic/manifest.csv \\
        --real-stats results/void_statistics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UDM Epic 1 — synthetic vs real distribution comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True, help="Synthetic manifest.csv")
    p.add_argument(
        "--real-stats",
        type=Path,
        default=None,
        help="JSON from scripts/analysis/analyze_real_voids.py",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Optional markdown report path")
    return p.parse_args()


def _load_real_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data: Dict[str, Any] = json.load(f)
    per = data.get("per_image", [])
    n_voids = np.array([float(x["n_voids"]) for x in per], dtype=np.float64)
    area = np.array([float(x["void_area_fraction"]) for x in per], dtype=np.float64)
    return n_voids, area


def _report_block(
    name: str,
    synth: np.ndarray,
    real: np.ndarray,
) -> str:
    ks_n, p_n = stats.ks_2samp(synth, real)
    lines = [
        f"### {name}",
        "",
        "| | synthetic | real |",
        "|--|--:|--:|",
        f"| n | {len(synth)} | {len(real)} |",
        f"| mean | {float(np.mean(synth)):.4f} | {float(np.mean(real)):.4f} |",
        f"| std | {float(np.std(synth)):.4f} | {float(np.std(real)):.4f} |",
        f"| KS statistic | {float(ks_n):.4f} | |",
        f"| p-value | {float(p_n):.4g} | |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.manifest)
    for col in ("n_voids", "total_void_area_fraction"):
        if col not in df.columns:
            print(f"Manifest missing column: {col}", file=sys.stderr)
            sys.exit(1)

    s_n = df["n_voids"].to_numpy(dtype=np.float64)
    s_a = df["total_void_area_fraction"].to_numpy(dtype=np.float64)

    print("Synthetic manifest:", args.manifest)
    print(f"  samples: {len(df)}")
    print(f"  n_voids: mean={s_n.mean():.4f} std={s_n.std():.4f}")
    print(f"  total_void_area_fraction: mean={s_a.mean():.4f} std={s_a.std():.4f}")

    if args.real_stats is None:
        print("\n(No --real-stats; skipping KS tests.)")
        return

    if not args.real_stats.is_file():
        print(f"Real stats not found: {args.real_stats}", file=sys.stderr)
        sys.exit(1)

    r_n, r_a = _load_real_arrays(args.real_stats)
    print("\nReal masks:", args.real_stats)
    print(f"  samples: {len(r_n)}")
    print(f"  n_voids: mean={r_n.mean():.4f} std={r_n.std():.4f}")
    print(f"  void_area_fraction: mean={r_a.mean():.4f} std={r_a.std():.4f}")

    report = "# Synthetic vs real — distribution comparison\n\n"
    report += _report_block("Void count (n_voids)", s_n, r_n)
    report += _report_block("Void area fraction", s_a, r_a)
    print("\n" + report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
