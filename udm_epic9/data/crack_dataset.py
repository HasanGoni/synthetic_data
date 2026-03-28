"""
Epic 9 — Dataset classes and generation utilities for synthetic crack data.

Provides a PyTorch :class:`~torch.utils.data.Dataset` for training crack
detection / segmentation models, and a bulk generation function that
writes images, masks, and a JSON manifest to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from udm_epic9.domain_transfer.usm_to_rgb import USMtoRGBTransfer, mask_to_image
from udm_epic9.models.crack_types import (
    delamination_crack,
    die_crack,
    mold_crack,
    substrate_crack,
)
from udm_epic9.rendering.usm_renderer import (
    _generate_usm_background,
    generate_synthetic_usm_with_cracks,
)

logger = logging.getLogger(__name__)

_CRACK_GENERATORS = {
    "die_crack": die_crack,
    "substrate_crack": substrate_crack,
    "mold_crack": mold_crack,
    "delamination_crack": delamination_crack,
}


class CrackDataset(Dataset):
    """PyTorch dataset for synthetic crack images and masks.

    If *images_dir* is ``None``, samples are generated on-the-fly.
    Otherwise, images and masks are loaded from disk.

    Parameters
    ----------
    images_dir : str or Path, optional
        Directory containing saved images.  If ``None``, generate on-the-fly.
    masks_dir : str or Path, optional
        Directory containing saved masks.
    crack_types : list[str], optional
        Crack types to generate (on-the-fly mode).
    image_size : tuple[int, int]
        ``(height, width)`` for output images.
    domain : str
        ``'usm'`` or ``'rgb'``.
    n_samples : int
        Number of samples when generating on-the-fly.
    empty_fraction : float
        Fraction of samples that contain no cracks.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        crack_types: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (512, 512),
        domain: str = "usm",
        n_samples: int = 100,
        empty_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.domain = domain
        self.crack_types = crack_types or list(_CRACK_GENERATORS.keys())
        self.empty_fraction = empty_fraction

        if images_dir is not None:
            self._mode = "disk"
            self._images_dir = Path(images_dir)
            self._masks_dir = Path(masks_dir) if masks_dir else None
            self._image_files = sorted(self._images_dir.glob("*.png"))
            if self._masks_dir:
                self._mask_files = sorted(self._masks_dir.glob("*.png"))
            else:
                self._mask_files = []
            self._n_samples = len(self._image_files)
        else:
            self._mode = "fly"
            self._n_samples = n_samples
            self._seed = seed

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._mode == "disk":
            return self._load_from_disk(idx)
        return self._generate_sample(idx)

    def _load_from_disk(self, idx: int) -> Dict[str, Any]:
        """Load a sample from saved files."""
        img_path = self._image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if self.domain == "usm" else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")

        h, w = self.image_size
        img = cv2.resize(img, (w, h))

        if self.domain == "usm":
            img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        mask_t = torch.zeros(1, h, w, dtype=torch.float32)
        crack_type = "unknown"
        if self._mask_files and idx < len(self._mask_files):
            mask = cv2.imread(str(self._mask_files[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return {
            "image": img_t,
            "mask": mask_t,
            "crack_type": crack_type,
            "domain": self.domain,
        }

    def _generate_sample(self, idx: int) -> Dict[str, Any]:
        """Generate a sample on-the-fly."""
        rng = np.random.default_rng(self._seed + idx)
        h, w = self.image_size

        # Decide if this sample should be empty (no cracks)
        is_empty = rng.random() < self.empty_fraction

        if is_empty:
            background = _generate_usm_background(h, w, rng=rng)
            mask = np.zeros((h, w), dtype=np.uint8)
            crack_type = "none"

            if self.domain == "rgb":
                transfer = USMtoRGBTransfer(method="colormap")
                image = transfer.transfer(background)
            else:
                image = background
        else:
            crack_type = str(rng.choice(self.crack_types))
            image, mask, _ = generate_synthetic_usm_with_cracks(
                height=h, width=w, n_cracks=1,
                crack_types=[crack_type], rng=rng,
            )
            if self.domain == "rgb":
                transfer = USMtoRGBTransfer(method="colormap")
                image = transfer.transfer_with_cracks(image, mask)

        # Convert to tensors
        if self.domain == "usm":
            img_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            img_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        mask_t = torch.from_numpy((mask > 127).astype(np.float32) if mask.max() > 1 else mask.astype(np.float32)).unsqueeze(0)

        return {
            "image": img_t,
            "mask": mask_t,
            "crack_type": crack_type,
            "domain": self.domain,
        }


def generate_crack_dataset(
    output_dir: str,
    n_samples: int = 1000,
    domains: Optional[List[str]] = None,
    config: Optional[dict] = None,
    seed: int = 42,
    image_size: Tuple[int, int] = (512, 512),
    empty_fraction: float = 0.1,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> Path:
    """Generate and save a full synthetic crack dataset with masks.

    Writes images and masks to *output_dir* organized by split and domain,
    and creates a JSON manifest file.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    n_samples : int
        Total number of samples to generate.
    domains : list[str], optional
        List of domains to generate (default ``['usm', 'rgb']``).
    config : dict, optional
        Override configuration parameters.
    seed : int
        Random seed.
    image_size : tuple[int, int]
        ``(height, width)`` of generated images.
    empty_fraction : float
        Fraction of no-crack images.
    train_ratio, val_ratio, test_ratio : float
        Dataset split ratios.

    Returns
    -------
    Path
        Path to the generated manifest JSON file.
    """
    if domains is None:
        domains = ["usm", "rgb"]

    if config is not None:
        n_samples = config.get("dataset", {}).get("n_samples", n_samples)
        seed = config.get("dataset", {}).get("seed", seed)
        domains = config.get("dataset", {}).get("domains", domains)
        empty_fraction = config.get("dataset", {}).get("empty_fraction", empty_fraction)
        train_ratio = config.get("dataset", {}).get("train_ratio", train_ratio)
        val_ratio = config.get("dataset", {}).get("val_ratio", val_ratio)
        test_ratio = config.get("dataset", {}).get("test_ratio", test_ratio)
        img_cfg = config.get("image", {})
        image_size = (img_cfg.get("height", image_size[0]), img_cfg.get("width", image_size[1]))

    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    h, w = image_size

    crack_types = list(_CRACK_GENERATORS.keys())
    transfer = USMtoRGBTransfer(method="colormap")

    # Determine split indices
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    splits = {}
    splits["train"] = indices[:n_train].tolist()
    splits["val"] = indices[n_train : n_train + n_val].tolist()
    splits["test"] = indices[n_train + n_val :].tolist()

    manifest_records: List[dict] = []

    for split_name, split_indices in splits.items():
        for domain in domains:
            img_dir = out / split_name / domain / "images"
            mask_dir = out / split_name / domain / "masks"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        sample_rng = np.random.default_rng(seed + i)
        is_empty = sample_rng.random() < empty_fraction

        if is_empty:
            usm_bg = _generate_usm_background(h, w, rng=sample_rng)
            mask = np.zeros((h, w), dtype=np.uint8)
            crack_type = "none"
        else:
            crack_type = str(sample_rng.choice(crack_types))
            usm_bg, mask, _ = generate_synthetic_usm_with_cracks(
                height=h, width=w, n_cracks=1,
                crack_types=[crack_type], rng=sample_rng,
            )

        # Determine split
        split_name = "train"
        for sn, si in splits.items():
            if i in si:
                split_name = sn
                break

        fname = f"{i:06d}.png"

        for domain in domains:
            if domain == "usm":
                img_save = np.clip(usm_bg * 255, 0, 255).astype(np.uint8)
            else:
                if is_empty:
                    img_save = transfer.transfer(usm_bg)
                    img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
                else:
                    img_save = transfer.transfer_with_cracks(usm_bg, mask)
                    img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)

            img_path = out / split_name / domain / "images" / fname
            mask_path = out / split_name / domain / "masks" / fname
            cv2.imwrite(str(img_path), img_save)
            cv2.imwrite(str(mask_path), mask)

        manifest_records.append({
            "index": i,
            "split": split_name,
            "crack_type": crack_type,
            "has_crack": not is_empty,
            "filename": fname,
        })

    manifest_path = out / "manifest.json"
    manifest = {
        "n_samples": n_samples,
        "domains": domains,
        "image_size": list(image_size),
        "seed": seed,
        "splits": {k: len(v) for k, v in splits.items()},
        "samples": manifest_records,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generated %d samples in %s", n_samples, out)
    return manifest_path
