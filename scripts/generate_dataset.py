#!/usr/bin/env python3
"""
generate_dataset.py — Run Epic 1 synthetic dataset generation.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --config configs/default.yaml --total 3000 --workers 8

This script is the primary entry point for running the full 3,000-image
generation pipeline on your Ubuntu desktop with RTX 2080 Ti.

Note: Generation uses CPU (joblib multiprocessing), not GPU.
GPU (RTX 2080 Ti) is used during UNet++ training in Epic 2.
"""

import sys
import argparse
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from udm_epic1.dataset.pipeline import DatasetPipeline, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="UDM Epic 1 — Synthetic Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--total", type=int, default=None, help="Override total image count")
    parser.add_argument("--workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Parse config, print plan, exit")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading config: {args.config}")
    cfg = PipelineConfig.from_yaml(args.config)

    if args.total:
        cfg.total_images = args.total
    if args.workers:
        cfg.num_workers = args.workers
    if args.output:
        cfg.output_dir = args.output

    print(f"Plan: {cfg.total_images} images @ {cfg.generator.height}×{cfg.generator.width}")
    print(f"      {cfg.num_workers} workers | output → {cfg.output_dir}")
    print(f"      Train {cfg.train_ratio:.0%} / Val {cfg.val_ratio:.0%} / Test {cfg.test_ratio:.0%}")

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    pipeline = DatasetPipeline(cfg)
    manifest = pipeline.run()
    print(f"\nManifest written to: {manifest}")


if __name__ == "__main__":
    main()
