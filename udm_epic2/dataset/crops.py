"""
Epic 2 US 2.1 — extract defect-centric crops from paired images and void masks.

Each connected component in the mask (values > 127) becomes one crop with optional padding.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from udm_epic2.conditioning.edges import edge_map_from_mask


@dataclass
class CropConfig:
    """Controls defect crop extraction."""

    min_component_area_px: int = 16
    """Minimum pixels in a void component to emit a crop."""

    padding_px: int = 8
    """Extra border around tight bounding box."""

    max_crop_side: int = 512
    """If max(H,W) exceeds this, crop is resized (image+mask together, same scale)."""

    defect_class: str = "void"
    """Label stored in manifest (extend when you have multi-class paths)."""


def _resize_pair_if_needed(
    img: np.ndarray,
    msk: np.ndarray,
    max_side: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resize image and mask together preserving alignment. Returns scale factor."""
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, msk, 1.0
    scale = max_side / float(side)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    msk_r = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r, scale


def extract_crops_for_pair(
    image: np.ndarray,
    mask: np.ndarray,
    source_id: str,
    cfg: CropConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Yield (crop_image, crop_mask, meta) for each void component in ``mask``.

    ``image`` may be 8-bit or 16-bit single channel or BGR; converted to single channel for bbox logic.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    bin_m = (mask > 127).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_m, connectivity=8)

    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < cfg.min_component_area_px:
            continue
        x, y, bw, bh = (
            int(stats[lab, cv2.CC_STAT_LEFT]),
            int(stats[lab, cv2.CC_STAT_TOP]),
            int(stats[lab, cv2.CC_STAT_WIDTH]),
            int(stats[lab, cv2.CC_STAT_HEIGHT]),
        )
        pad = cfg.padding_px
        ih, iw = image.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(iw, x + bw + pad)
        y1 = min(ih, y + bh + pad)

        crop_img = np.asarray(image[y0:y1, x0:x1]).copy()
        crop_msk = bin_m[y0:y1, x0:x1].astype(np.uint8) * 255

        crop_img, crop_msk, sc = _resize_pair_if_needed(crop_img, crop_msk, cfg.max_crop_side)

        meta: Dict[str, Any] = {
            "source_id": source_id,
            "defect_class": cfg.defect_class,
            "component_label": lab,
            "bbox_xyxy_orig": (x0, y0, x1, y1),
            "component_area_px": area,
            "resize_scale": sc,
        }
        yield crop_img, crop_msk, meta


def process_crop_dataset(
    image_dir: Path,
    mask_dir: Path,
    out_root: Path,
    cfg: CropConfig,
    glob: str = "*.png",
    image_subdir: str = "images",
    mask_subdir: str = "masks",
    edge_subdir: str = "edges",
    write_edges: bool = False,
) -> Path:
    """
    Walk ``image_dir`` for ``glob``; expect matching mask file in ``mask_dir``.

    Writes:
      ``out_root / image_subdir / {defect_class} / {crop_id}.png``
      ``out_root / mask_subdir / {defect_class} / {crop_id}.png``
      ``out_root / edge_subdir / {defect_class} / {crop_id}.png`` (optional)

    Returns path to ``out_root / manifest.csv``.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    out_root = Path(out_root)
    img_out = out_root / image_subdir / cfg.defect_class
    msk_out = out_root / mask_subdir / cfg.defect_class
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)
    edg_out: Optional[Path] = None
    if write_edges:
        edg_out = out_root / edge_subdir / cfg.defect_class
        edg_out.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob(glob))
    rows: List[Dict[str, Any]] = []
    crop_idx = 0

    for ip in tqdm(paths, desc="Epic2 crops", unit="img"):
        if not ip.is_file():
            continue
        mp = mask_dir / ip.name
        if not mp.is_file():
            continue
        image = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        if image.shape[:2] != mask.shape[:2]:
            continue

        stem = ip.stem
        for crop_img, crop_msk, meta in extract_crops_for_pair(image, mask, stem, cfg):
            crop_id = f"{stem}_c{meta['component_label']}_{crop_idx:05d}"
            crop_idx += 1
            meta["crop_id"] = crop_id
            meta["image_relpath"] = f"{image_subdir}/{cfg.defect_class}/{crop_id}.png"
            meta["mask_relpath"] = f"{mask_subdir}/{cfg.defect_class}/{crop_id}.png"

            out_i = img_out / f"{crop_id}.png"
            out_m = msk_out / f"{crop_id}.png"
            cv2.imwrite(str(out_i), crop_img)
            cv2.imwrite(str(out_m), crop_msk)

            if write_edges and edg_out is not None:
                edge = edge_map_from_mask(crop_msk)
                out_e = edg_out / f"{crop_id}.png"
                cv2.imwrite(str(out_e), edge)
                meta["edge_relpath"] = f"{edge_subdir}/{cfg.defect_class}/{crop_id}.png"
            rows.append(meta)

    manifest = out_root / "manifest.csv"
    if not rows:
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["crop_id", "source_id", "defect_class"])
            w.writeheader()
        return manifest

    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    return manifest
