"""
Epic 3 — PyTorch Datasets for CycleGAN cross-modality translation.

Provides two dataset classes:

* :class:`UnpairedDataset` — loads images from two independent modality
  directories (e.g. AOI and USM) and pairs them randomly.  Used for
  CycleGAN training where paired supervision is not available.

* :class:`PairedDataset` — loads registered image pairs (same filenames
  in both directories) with optional masks.  Used for validation and
  testing where ground-truth correspondences exist.

All images are loaded as single-channel grayscale and normalised to
``[-1, 1]`` to match the ``Tanh`` output range of the generator.

Usage::

    from udm_epic3.data.unpaired_dataset import UnpairedDataset

    ds = UnpairedDataset(dir_A="data/aoi/train", dir_B="data/usm/train",
                         image_size=(512, 512))
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Supported image file extensions (case-insensitive matching).
_IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".bmp"}


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------


def _collect_image_paths(directory: Path) -> list[Path]:
    """Return sorted list of image file paths found in *directory*."""
    if not directory.is_dir():
        logger.warning("Image directory does not exist: %s", directory)
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _load_grayscale(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load an image as grayscale uint8, resize to ``(H, W)``."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if (img.shape[0], img.shape[1]) != image_size:
        img = cv2.resize(
            img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR,
        )
    return img


def _to_tensor_tanh(img: np.ndarray) -> torch.Tensor:
    """Convert grayscale uint8 HxW to float32 ``[1, H, W]`` in ``[-1, 1]``.

    Normalisation: ``pixel / 127.5 - 1.0`` maps ``[0, 255]`` to ``[-1, 1]``.
    """
    img_f = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img_f).unsqueeze(0)  # (1, H, W)


def _load_mask(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Load a mask as grayscale, resize (nearest), binarise, return ``[1, H, W]``."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if (mask.shape[0], mask.shape[1]) != image_size:
        mask = cv2.resize(
            mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask > 127).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)  # (1, H, W)


# ------------------------------------------------------------------
# UnpairedDataset
# ------------------------------------------------------------------


class UnpairedDataset(Dataset):
    """Unpaired image dataset for CycleGAN training.

    Loads images from two modality folders (e.g. AOI and USM) and pairs
    them randomly.  Domain B is indexed via a shuffled index list so
    that each epoch presents a different random pairing.

    Parameters
    ----------
    dir_A : str | Path
        Directory containing domain-A images (e.g. AOI).
    dir_B : str | Path
        Directory containing domain-B images (e.g. USM).
    image_size : tuple[int, int]
        Target ``(height, width)`` for resizing.  Default ``(512, 512)``.
    transform : callable | None
        Optional transform applied to the dict
        ``{'A': Tensor, 'B': Tensor, ...}`` and returning the same structure.
    masks_A : str | Path | None
        Optional masks directory for domain A (defect-aware training).
    masks_B : str | Path | None
        Optional masks directory for domain B.
    """

    def __init__(
        self,
        dir_A: Union[str, Path],
        dir_B: Union[str, Path],
        image_size: tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        masks_A: Union[str, Path, None] = None,
        masks_B: Union[str, Path, None] = None,
    ) -> None:
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.image_size = image_size
        self.transform = transform
        self.masks_A_dir = Path(masks_A) if masks_A is not None else None
        self.masks_B_dir = Path(masks_B) if masks_B is not None else None

        self.paths_A = _collect_image_paths(self.dir_A)
        self.paths_B = _collect_image_paths(self.dir_B)

        if len(self.paths_A) == 0:
            logger.warning("No images found in dir_A: %s", self.dir_A)
        if len(self.paths_B) == 0:
            logger.warning("No images found in dir_B: %s", self.dir_B)

        # Pre-compute a shuffled index order for domain B so that unpaired
        # sampling yields different random pairings each time.
        self._b_indices = list(range(len(self.paths_B))) if self.paths_B else []
        random.shuffle(self._b_indices)

        # Build mask stem->path lookups for fast access.
        self._masks_A_lookup: dict[str, Path] = {}
        self._masks_B_lookup: dict[str, Path] = {}
        if self.masks_A_dir is not None and self.masks_A_dir.is_dir():
            for p in self.masks_A_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    self._masks_A_lookup[p.stem] = p
        if self.masks_B_dir is not None and self.masks_B_dir.is_dir():
            for p in self.masks_B_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    self._masks_B_lookup[p.stem] = p

        logger.info(
            "UnpairedDataset  A=%d  B=%d  masks_A=%d  masks_B=%d  size=%s",
            len(self.paths_A), len(self.paths_B),
            len(self._masks_A_lookup), len(self._masks_B_lookup),
            self.image_size,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Length is determined by the larger domain (wrap the smaller one)."""
        return max(len(self.paths_A), len(self.paths_B), 1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Domain A — index with wrap-around
        idx_a = index % len(self.paths_A) if self.paths_A else 0
        path_a = self.paths_A[idx_a]
        img_a = _load_grayscale(path_a, self.image_size)
        tensor_a = _to_tensor_tanh(img_a)

        # Domain B — shuffled index with wrap-around for random pairing
        idx_b = self._b_indices[index % len(self._b_indices)] if self._b_indices else 0
        path_b = self.paths_B[idx_b]
        img_b = _load_grayscale(path_b, self.image_size)
        tensor_b = _to_tensor_tanh(img_b)

        sample: Dict[str, Any] = {
            "A": tensor_a,
            "B": tensor_b,
            "path_A": str(path_a),
            "path_B": str(path_b),
        }

        # Load masks if directories were provided.
        mask_a_path = self._masks_A_lookup.get(path_a.stem)
        if mask_a_path is not None:
            sample["mask_A"] = _load_mask(mask_a_path, self.image_size)
        elif self.masks_A_dir is not None:
            sample["mask_A"] = torch.zeros(1, *self.image_size)

        mask_b_path = self._masks_B_lookup.get(path_b.stem)
        if mask_b_path is not None:
            sample["mask_B"] = _load_mask(mask_b_path, self.image_size)
        elif self.masks_B_dir is not None:
            sample["mask_B"] = torch.zeros(1, *self.image_size)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __repr__(self) -> str:
        return (
            f"UnpairedDataset(dir_A='{self.dir_A}', dir_B='{self.dir_B}', "
            f"n_A={len(self.paths_A)}, n_B={len(self.paths_B)})"
        )


# ------------------------------------------------------------------
# PairedDataset
# ------------------------------------------------------------------


class PairedDataset(Dataset):
    """Paired image dataset for CycleGAN evaluation.

    Loads registered image pairs (same filenames in both directories)
    with optional binary masks for defect-preservation evaluation.

    Parameters
    ----------
    dir_A : str | Path
        Directory containing domain-A images.
    dir_B : str | Path
        Directory containing domain-B images with matching filenames.
    masks_A : str | Path | None
        Optional directory of binary masks for domain A.
    masks_B : str | Path | None
        Optional directory of binary masks for domain B.
    image_size : tuple[int, int]
        Target ``(height, width)`` for resizing.  Default ``(512, 512)``.
    """

    def __init__(
        self,
        dir_A: Union[str, Path],
        dir_B: Union[str, Path],
        masks_A: Union[str, Path, None] = None,
        masks_B: Union[str, Path, None] = None,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.masks_A_dir = Path(masks_A) if masks_A is not None else None
        self.masks_B_dir = Path(masks_B) if masks_B is not None else None
        self.image_size = image_size

        # Find matching filenames (by stem) present in both directories.
        paths_a = _collect_image_paths(self.dir_A)
        paths_b = _collect_image_paths(self.dir_B)

        stems_b: dict[str, Path] = {p.stem: p for p in paths_b}
        self.pairs: list[tuple[Path, Path]] = []
        for pa in paths_a:
            if pa.stem in stems_b:
                self.pairs.append((pa, stems_b[pa.stem]))
        self.pairs.sort(key=lambda t: t[0].stem)

        # Build mask stem->path lookups.
        self._masks_A_lookup: dict[str, Path] = {}
        self._masks_B_lookup: dict[str, Path] = {}
        if self.masks_A_dir is not None and self.masks_A_dir.is_dir():
            for p in self.masks_A_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    self._masks_A_lookup[p.stem] = p
        if self.masks_B_dir is not None and self.masks_B_dir.is_dir():
            for p in self.masks_B_dir.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    self._masks_B_lookup[p.stem] = p

        if len(self.pairs) == 0:
            logger.warning(
                "No matching filename pairs found between %s and %s",
                self.dir_A, self.dir_B,
            )

        logger.info(
            "PairedDataset  pairs=%d  masks_A=%d  masks_B=%d  size=%s",
            len(self.pairs),
            len(self._masks_A_lookup),
            len(self._masks_B_lookup),
            self.image_size,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path_a, path_b = self.pairs[index]
        filename = path_a.stem

        # Images
        img_a = _load_grayscale(path_a, self.image_size)
        img_b = _load_grayscale(path_b, self.image_size)
        tensor_a = _to_tensor_tanh(img_a)
        tensor_b = _to_tensor_tanh(img_b)

        # Masks (optional)
        mask_a: Optional[torch.Tensor] = None
        mask_b: Optional[torch.Tensor] = None

        mask_a_path = self._masks_A_lookup.get(filename)
        if mask_a_path is not None:
            mask_a = _load_mask(mask_a_path, self.image_size)

        mask_b_path = self._masks_B_lookup.get(filename)
        if mask_b_path is not None:
            mask_b = _load_mask(mask_b_path, self.image_size)

        return {
            "A": tensor_a,
            "B": tensor_b,
            "mask_A": mask_a,
            "mask_B": mask_b,
            "filename": filename,
        }

    def __repr__(self) -> str:
        return (
            f"PairedDataset(dir_A='{self.dir_A}', dir_B='{self.dir_B}', "
            f"n_pairs={len(self.pairs)})"
        )


# ------------------------------------------------------------------
# Factory: build datasets from config dict
# ------------------------------------------------------------------


def build_cyclegan_datasets(config: dict) -> dict[str, Any]:
    """Construct CycleGAN datasets from a YAML configuration dict.

    Expected config schema::

        data:
          image_size: [512, 512]
          train:
            dir_A: /path/to/aoi/train
            dir_B: /path/to/usm/train
            masks_A: /path/to/aoi/train_masks   # optional
            masks_B: /path/to/usm/train_masks   # optional
          val:                    # optional
            dir_A: /path/to/aoi/val
            dir_B: /path/to/usm/val
            masks_A: /path/to/aoi/val_masks     # optional
            masks_B: /path/to/usm/val_masks     # optional
          test:                   # optional
            dir_A: /path/to/aoi/test
            dir_B: /path/to/usm/test
            masks_A: /path/to/aoi/test_masks    # optional
            masks_B: /path/to/usm/test_masks    # optional

    Parameters
    ----------
    config : dict
        The full YAML configuration dict (must contain a ``data`` key).

    Returns
    -------
    dict
        Keys: ``train`` (:class:`UnpairedDataset`),
        ``val`` (:class:`PairedDataset` | None),
        ``test`` (:class:`PairedDataset` | None).
    """
    data_cfg = config.get("data", config)
    image_size = tuple(data_cfg.get("image_size", [512, 512]))

    # --- Training set (unpaired) ---
    train_cfg = data_cfg["train"]
    train_ds = UnpairedDataset(
        dir_A=train_cfg["dir_A"],
        dir_B=train_cfg["dir_B"],
        image_size=image_size,
        masks_A=train_cfg.get("masks_A"),
        masks_B=train_cfg.get("masks_B"),
    )

    # --- Validation set (paired, optional) ---
    val_ds: Optional[PairedDataset] = None
    val_cfg = data_cfg.get("val")
    if val_cfg is not None:
        val_ds = PairedDataset(
            dir_A=val_cfg["dir_A"],
            dir_B=val_cfg["dir_B"],
            masks_A=val_cfg.get("masks_A"),
            masks_B=val_cfg.get("masks_B"),
            image_size=image_size,
        )

    # --- Test set (paired, optional) ---
    test_ds: Optional[PairedDataset] = None
    test_cfg = data_cfg.get("test")
    if test_cfg is not None:
        test_ds = PairedDataset(
            dir_A=test_cfg["dir_A"],
            dir_B=test_cfg["dir_B"],
            masks_A=test_cfg.get("masks_A"),
            masks_B=test_cfg.get("masks_B"),
            image_size=image_size,
        )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }
