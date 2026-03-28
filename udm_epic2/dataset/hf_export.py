"""
Export Epic 2 crop folder to a HuggingFace-compatible folder for ``train_controlnet.py``.

Layout written::

    out_dir/
      train/
        image/0000.png ...
        conditioning_image/0000.png ...
      metadata.csv

``metadata.csv`` columns: ``file_name``, ``text``, ``conditioning_file``
(compatible with diffusers controlnet training when using a custom loading script).

For in-repo training we read paths directly from ``manifest.csv`` produced by ``process_crop_dataset``.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd


def export_hf_style_folder(
    epic2_crops_root: Path,
    out_dir: Path,
    caption: str = "semiconductor x-ray void defect, high contrast",
    manifest_name: str = "manifest.csv",
) -> Path:
    """
    Copy ``images/{class}/`` and ``edges/{class}/`` pairs into ``out_dir/train/`` with aligned names.

    Returns path to ``out_dir/train/metadata.csv``.
    """
    epic2_crops_root = Path(epic2_crops_root)
    man = epic2_crops_root / manifest_name
    if not man.is_file():
        raise FileNotFoundError(f"manifest not found: {man}")

    df = pd.read_csv(man)
    if "image_relpath" not in df.columns or "edge_relpath" not in df.columns:
        raise ValueError("manifest must contain image_relpath and edge_relpath (run extract with write_edges=true)")

    train_root = Path(out_dir) / "train"
    img_d = train_root / "image"
    cond_d = train_root / "conditioning_image"
    img_d.mkdir(parents=True, exist_ok=True)
    cond_d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in enumerate(df.to_dict("records")):
        src_i = epic2_crops_root / str(r["image_relpath"])
        src_e = epic2_crops_root / str(r["edge_relpath"])
        if not src_i.is_file() or not src_e.is_file():
            continue
        name = f"{i:06d}.png"
        shutil.copy2(src_i, img_d / name)
        shutil.copy2(src_e, cond_d / name)
        rows.append(
            {
                "file_name": f"image/{name}",
                "text": caption,
                "conditioning_file": f"conditioning_image/{name}",
            }
        )

    meta = train_root / "metadata.csv"
    with open(meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text", "conditioning_file"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    return meta
