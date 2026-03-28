#!/usr/bin/env python3
"""
Analyze real void masks: per-image metrics + JSON summary (Epic 1 Script 1.2A-style).

Example:
    python scripts/analysis/analyze_real_voids.py \\
        --mask-dir data/real/train/masks \\
        --output results/void_statistics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from udm_epic1.validation.mask_stats import scan_mask_directory, write_stats_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UDM Epic 1 — aggregate statistics from real void masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mask-dir", required=True, type=Path, help="Directory of binary mask PNGs")
    p.add_argument("--glob", default="*.png", help="Filename glob under mask-dir")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mask_dir.is_dir():
        print(f"Not a directory: {args.mask_dir}", file=sys.stderr)
        sys.exit(1)

    records, summary = scan_mask_directory(args.mask_dir, glob=args.glob)
    if not records:
        print(f"No masks matched {args.glob!r} under {args.mask_dir}", file=sys.stderr)
        sys.exit(1)

    write_stats_json(records, summary, args.output)
    print(f"Wrote {len(records)} masks → {args.output}")
    s = summary.get("void_area_fraction", {})
    nvoid = summary.get("n_voids", {})
    print(
        f"  void_area_fraction: mean={s.get('mean', 0):.4f}  "
        f"n_voids: mean={nvoid.get('mean', 0):.2f}"
    )


if __name__ == "__main__":
    main()
