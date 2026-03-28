"""PyTorch Dataset for multi-site domain-adaptive segmentation.

Loads grayscale X-ray images (and optional binary masks) from disk,
assigns a domain label to each sample, and applies optional transforms.
Also provides a factory function to build train/val/target datasets
from a YAML configuration dict.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (used even for grayscale→3-ch images so
# that pretrained encoder statistics are consistent).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Supported image file extensions (case-insensitive matching).
_IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".bmp"}


def _collect_image_paths(directory: Path) -> list[Path]:
    """Return sorted list of image file paths found in *directory*."""
    paths = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    return paths


def _load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image as grayscale, resize, and return as uint8 HxW array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if (img.shape[0], img.shape[1]) != image_size:
        img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    return img


def _normalise_image(img: np.ndarray) -> torch.Tensor:
    """Convert grayscale uint8 HxW to normalised float32 [3,H,W] tensor.

    Steps:
        1. Scale to [0, 1].
        2. Repeat across 3 channels.
        3. Normalise per-channel with ImageNet mean/std.
    """
    img_f = img.astype(np.float32) / 255.0
    # Stack to (H, W, 3)
    img_3ch = np.stack([img_f, img_f, img_f], axis=-1)
    # Normalise
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_3ch = (img_3ch - mean) / std
    # HWC -> CHW
    tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1))
    return tensor


def _load_mask(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Load a mask as grayscale, resize, binarise, and return float32 [1,H,W]."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if (mask.shape[0], mask.shape[1]) != image_size:
        mask = cv2.resize(mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)  # (1, H, W)


class DomainDataset(Dataset):
    """Dataset that pairs images with optional masks and a domain label.

    Parameters
    ----------
    images_dir : str | Path
        Directory containing image files (png, tif, tiff, jpg, bmp).
    masks_dir : str | Path | None
        Directory containing corresponding mask files.  Filenames must match
        the image filenames (same stem).  ``None`` for unlabeled target domains.
    domain_label : int
        Integer label identifying the acquisition domain / site.
    transform : callable | None
        Optional transform applied to the **image tensor** (after
        normalisation) and **mask tensor** jointly.  The callable receives
        a dict ``{'image': Tensor, 'mask': Tensor | None}`` and must return
        the same structure.
    image_size : tuple[int, int]
        Target (height, width) for resizing.  Default ``(512, 512)``.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path, None] = None,
        domain_label: int = 0,
        transform: Optional[Callable] = None,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.domain_label = domain_label
        self.transform = transform
        self.image_size = image_size

        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"images_dir does not exist: {self.images_dir}")

        self.image_paths = _collect_image_paths(self.images_dir)
        if len(self.image_paths) == 0:
            logger.warning("No images found in %s", self.images_dir)

        # Pre-build mask path lookup (stem -> path) for fast access.
        self._mask_lookup: dict[str, Path] = {}
        if self.masks_dir is not None:
            if not self.masks_dir.is_dir():
                raise NotADirectoryError(f"masks_dir does not exist: {self.masks_dir}")
            for p in self.masks_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    self._mask_lookup[p.stem] = p

        logger.info(
            "DomainDataset domain=%d  images=%d  masks=%d  size=%s",
            self.domain_label,
            len(self.image_paths),
            len(self._mask_lookup),
            self.image_size,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_path = self.image_paths[index]

        # --- Image ---
        img_np = _load_image(img_path, self.image_size)
        image_tensor = _normalise_image(img_np)

        # --- Mask ---
        mask_tensor: Optional[torch.Tensor] = None
        if self.masks_dir is not None:
            mask_path = self._mask_lookup.get(img_path.stem)
            if mask_path is not None:
                mask_tensor = _load_mask(mask_path, self.image_size)
            else:
                logger.warning("Mask not found for image %s", img_path.name)

        # --- Optional transform ---
        if self.transform is not None:
            transformed = self.transform({"image": image_tensor, "mask": mask_tensor})
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "domain": self.domain_label,
            "path": str(img_path),
        }

    def __repr__(self) -> str:
        return (
            f"DomainDataset(images_dir='{self.images_dir}', "
            f"domain={self.domain_label}, n={len(self)})"
        )


# ----------------------------------------------------------------------
# Factory: build datasets from YAML config dict
# ----------------------------------------------------------------------

def build_datasets_from_config(config: dict) -> dict[str, Any]:
    """Construct datasets from the ``data`` section of a YAML config.

    Expected config schema::

        data:
          image_size: [512, 512]
          source:
            images_dir: /path/to/source/images
            masks_dir: /path/to/source/masks
            domain_label: 0
            train_ratio: 0.8          # fraction for training split
          targets:
            - name: site_b
              images_dir: /path/to/site_b/images
              masks_dir: null          # unlabeled
              domain_label: 1
          evaluation:                  # optional; labeled target sets for eval
            - name: site_b_eval
              images_dir: /path/to/site_b_eval/images
              masks_dir: /path/to/site_b_eval/masks
              domain_label: 1

    Parameters
    ----------
    config : dict
        The ``data`` section of the full YAML configuration.

    Returns
    -------
    dict
        Keys: ``source_train``, ``source_val``, ``targets`` (list),
        ``evaluation`` (list).  Each value is a :class:`DomainDataset`.
    """
    image_size = tuple(config.get("image_size", [512, 512]))

    # --- Source domain (train / val split) ---
    src_cfg = config["source"]
    full_source = DomainDataset(
        images_dir=src_cfg["images_dir"],
        masks_dir=src_cfg.get("masks_dir"),
        domain_label=src_cfg.get("domain_label", 0),
        image_size=image_size,
    )

    train_ratio = float(src_cfg.get("train_ratio", 0.8))
    n_total = len(full_source)
    n_train = max(1, math.floor(n_total * train_ratio))
    n_val = n_total - n_train

    # Deterministic split using a fixed generator for reproducibility.
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_source, [n_train, n_val], generator=generator
    )

    # Wrap subsets so they still carry the DomainDataset interface attributes
    # downstream code may inspect (e.g. domain_label, image_paths length).
    # We attach convenience attributes but the subsets are valid Datasets.
    train_subset.domain_label = full_source.domain_label  # type: ignore[attr-defined]
    val_subset.domain_label = full_source.domain_label  # type: ignore[attr-defined]

    logger.info("Source split: train=%d  val=%d", n_train, n_val)

    # --- Target domains (unlabeled) ---
    target_datasets: list[DomainDataset] = []
    for tgt_cfg in config.get("targets", []):
        ds = DomainDataset(
            images_dir=tgt_cfg["images_dir"],
            masks_dir=tgt_cfg.get("masks_dir"),
            domain_label=tgt_cfg.get("domain_label", 1),
            image_size=image_size,
        )
        target_datasets.append(ds)
        logger.info("Target '%s': %d images", tgt_cfg.get("name", "unnamed"), len(ds))

    # --- Evaluation sets (labeled target domains for metrics) ---
    eval_datasets: list[DomainDataset] = []
    for eval_cfg in config.get("evaluation", []):
        ds = DomainDataset(
            images_dir=eval_cfg["images_dir"],
            masks_dir=eval_cfg.get("masks_dir"),
            domain_label=eval_cfg.get("domain_label", 1),
            image_size=image_size,
        )
        eval_datasets.append(ds)
        logger.info("Evaluation '%s': %d images", eval_cfg.get("name", "unnamed"), len(ds))

    return {
        "source_train": train_subset,
        "source_val": val_subset,
        "targets": target_datasets,
        "evaluation": eval_datasets,
    }
