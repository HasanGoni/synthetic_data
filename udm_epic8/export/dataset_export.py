"""
Epic 8 -- Dataset export to COCO, YOLO and HuggingFace formats.

Each function reads the images + masks produced by the unified pipeline and
writes the annotations / directory layout expected by the target ecosystem.

Functions
---------
export_to_coco   Convert to COCO instance segmentation JSON.
export_to_yolo   Convert to YOLO segmentation TXT labels.
export_to_hf     Convert to HuggingFace ``datasets`` arrow format.
merge_datasets   Merge multiple modality directories into one.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── COCO export ───────────────────────────────────────────────────────────────


def export_to_coco(
    data_dir: str,
    output_path: str,
    modality: Optional[str] = None,
) -> Path:
    """Convert a synthetic dataset directory to COCO instance segmentation JSON.

    The function scans ``data_dir/images`` for PNG files and looks for
    matching masks in ``data_dir/masks``.  Binary masks are converted to
    RLE-style polygon annotations.

    Parameters
    ----------
    data_dir : str
        Root of the modality dataset (must contain ``images/`` and ``masks/``).
    output_path : str
        Path where the COCO JSON will be written.
    modality : str, optional
        If given, written into the ``info`` block of the JSON.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    coco: Dict[str, Any] = {
        "info": {
            "description": f"UDM Epic 8 synthetic dataset ({modality or 'unknown'})",
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "defect", "supercategory": "defect"},
        ],
    }

    image_files = sorted(images_dir.glob("*.png"))
    ann_id = 1

    for img_id, img_path in enumerate(image_files, start=1):
        from PIL import Image

        img = Image.open(img_path)
        w, h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Look for a corresponding mask
        mask_name = img_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            # Try exact name match
            mask_path = masks_dir / img_path.name
        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path).convert("L"))
            bbox, area, segmentation = _mask_to_coco_annotation(mask_arr)
            if area > 0:
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                })
                ann_id += 1

    output_path.write_text(json.dumps(coco, indent=2))
    logger.info("COCO JSON written with %d images, %d annotations -> %s",
                len(coco["images"]), len(coco["annotations"]), output_path)
    return output_path


def _mask_to_coco_annotation(
    mask: np.ndarray,
    threshold: int = 127,
) -> tuple[list, float, list]:
    """Extract bbox, area and polygon segmentation from a binary mask.

    Returns ``([x, y, w, h], area, [[x1,y1,x2,y2,...]])``.
    """
    binary = (mask > threshold).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0], 0.0, []

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    area = float(binary.sum())

    # Simple bounding-box polygon as a baseline segmentation
    segmentation = [[
        x_min, y_min,
        x_max, y_min,
        x_max, y_max,
        x_min, y_max,
    ]]
    return bbox, area, segmentation


# ── YOLO export ───────────────────────────────────────────────────────────────


def export_to_yolo(data_dir: str, output_path: str) -> Path:
    """Convert dataset to YOLO segmentation format.

    Produces ``labels/*.txt`` files with normalised bounding-box annotations
    (class x_center y_center width height) and writes a ``data.yaml``
    describing the dataset.

    Parameters
    ----------
    data_dir : str
        Root of the modality dataset.
    output_path : str
        Output directory for the YOLO-formatted dataset.

    Returns
    -------
    Path
        The *output_path* directory.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    images_out = output_path / "images"
    labels_out = output_path / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted(images_dir.glob("*.png"))
    for img_path in image_files:
        # Copy image
        shutil.copy2(img_path, images_out / img_path.name)

        # Generate label
        mask_name = img_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            mask_path = masks_dir / img_path.name

        label_path = labels_out / (img_path.stem + ".txt")
        if mask_path.exists():
            from PIL import Image

            mask_arr = np.array(Image.open(mask_path).convert("L"))
            h, w = mask_arr.shape
            binary = (mask_arr > 127).astype(np.uint8)
            ys, xs = np.where(binary > 0)
            if len(xs) > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                x_center = ((x_min + x_max) / 2.0) / w
                y_center = ((y_min + y_max) / 2.0) / h
                bw = (x_max - x_min + 1) / w
                bh = (y_max - y_min + 1) / h
                label_path.write_text(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                label_path.write_text("")
        else:
            label_path.write_text("")

    # Write data.yaml
    data_yaml = output_path / "data.yaml"
    data_yaml.write_text(
        "names:\n"
        "  0: defect\n"
        f"nc: 1\n"
        f"train: {images_out}\n"
        f"val: {images_out}\n"
    )
    logger.info("YOLO export: %d images -> %s", len(image_files), output_path)
    return output_path


# ── HuggingFace export ────────────────────────────────────────────────────────


def export_to_hf(data_dir: str, output_path: str) -> Path:
    """Convert dataset to HuggingFace ``datasets`` directory layout.

    Creates a directory with ``train/`` and ``metadata.jsonl`` suitable for
    ``datasets.load_dataset("imagefolder", data_dir=...)``.

    Parameters
    ----------
    data_dir : str
        Root of the modality dataset.
    output_path : str
        Output directory for the HF-formatted dataset.

    Returns
    -------
    Path
        The *output_path* directory.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted(images_dir.glob("*.png"))
    metadata_lines: list[str] = []

    for img_path in image_files:
        shutil.copy2(img_path, train_dir / img_path.name)

        mask_name = img_path.stem + "_mask.png"
        mask_path = masks_dir / mask_name
        has_mask = mask_path.exists()
        if has_mask:
            shutil.copy2(mask_path, train_dir / mask_name)

        entry = {
            "file_name": img_path.name,
            "mask_name": mask_name if has_mask else None,
            "label": "defect",
        }
        metadata_lines.append(json.dumps(entry))

    metadata_path = output_path / "metadata.jsonl"
    metadata_path.write_text("\n".join(metadata_lines) + "\n")
    logger.info("HF export: %d images -> %s", len(image_files), output_path)
    return output_path


# ── Merge datasets ────────────────────────────────────────────────────────────


def merge_datasets(dirs: List[str], output_dir: str) -> Path:
    """Merge multiple modality dataset directories into one combined dataset.

    Images and masks are copied into a single ``images/`` and ``masks/``
    directory, prefixed by the source directory name to avoid collisions.
    A combined manifest is written.

    Parameters
    ----------
    dirs : list of str
        Paths to modality dataset directories, each containing ``images/``
        and ``masks/``.
    output_dir : str
        Destination for the merged dataset.

    Returns
    -------
    Path
        Path to the merged manifest JSON.
    """
    output_dir = Path(output_dir)
    merged_images = output_dir / "images"
    merged_masks = output_dir / "masks"
    merged_images.mkdir(parents=True, exist_ok=True)
    merged_masks.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []
    total = 0

    for src_dir_str in dirs:
        src_dir = Path(src_dir_str)
        prefix = src_dir.name
        img_dir = src_dir / "images"
        mask_dir = src_dir / "masks"

        if not img_dir.is_dir():
            logger.warning("Skipping %s: no images/ directory found", src_dir)
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            dest_name = f"{prefix}_{img_path.name}"
            shutil.copy2(img_path, merged_images / dest_name)

            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                shutil.copy2(mask_path, merged_masks / f"{prefix}_{mask_name}")

            manifest_entries.append({
                "image": dest_name,
                "source": prefix,
            })
            total += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "merged": True,
        "sources": [Path(d).name for d in dirs],
        "total_images": total,
        "entries": manifest_entries,
    }, indent=2))
    logger.info("Merged %d images from %d sources -> %s", total, len(dirs), output_dir)
    return manifest_path
