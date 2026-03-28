"""
PyTorch Dataset for Chromasense multi-spectral defect detection.

Provides :class:`SpectralDataset` which generates or loads multi-spectral
images with defect masks and labels on the fly, and
:func:`generate_spectral_dataset` which pre-generates and saves a dataset
to disk as ``.npz`` files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from udm_epic7.spectral.wavelength_model import SpectralConfig, default_spectral_config
from udm_epic7.rendering.spectral_renderer import render_spectral_image

logger = logging.getLogger(__name__)

# Defect types and their generation probabilities.
_DEFECT_TYPES = ["delamination", "contamination", "oxidation"]
_CONTAMINANTS = ["flux_residue", "organic_film", "dust", "solder_splash"]
_MATERIALS = ["silicon", "copper", "solder"]


class SpectralDataset(Dataset):
    """PyTorch Dataset that generates synthetic multi-spectral images on the fly.

    Each sample is a dict with:
        - ``image``: ``Tensor[C, H, W]`` (C = number of wavelengths)
        - ``mask``: ``Tensor[1, H, W]`` (binary defect mask)
        - ``defect_type``: ``str`` (``"delamination"`` | ``"contamination"`` | ``"oxidation"`` | ``"none"``)

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    config : SpectralConfig | None
        Spectral configuration.
    height, width : int
        Image spatial dimensions.
    defect_prob : float
        Probability that a sample contains a defect.
    seed : int
        Random seed for reproducibility.
    from_dir : str | Path | None
        If given, load pre-generated samples from this directory instead
        of generating on the fly.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        config: Optional[SpectralConfig] = None,
        height: int = 512,
        width: int = 512,
        defect_prob: float = 0.7,
        seed: int = 42,
        from_dir: Optional[str] = None,
    ) -> None:
        self.n_samples = n_samples
        self.config = config or default_spectral_config()
        self.height = height
        self.width = width
        self.defect_prob = defect_prob
        self.seed = seed
        self.from_dir = Path(from_dir) if from_dir else None

        # If loading from disk, discover available files
        self._file_list: Optional[List[Path]] = None
        if self.from_dir is not None and self.from_dir.is_dir():
            self._file_list = sorted(self.from_dir.glob("sample_*.npz"))
            self.n_samples = len(self._file_list)
            logger.info("Loaded %d samples from %s", self.n_samples, self.from_dir)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self._file_list is not None:
            return self._load_from_disk(index)
        return self._generate_sample(index)

    def _generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single sample procedurally."""
        rng = np.random.default_rng(self.seed + index)

        # Create random material layout
        layout = _random_layout(self.height, self.width, rng)

        # Decide whether to add a defect
        has_defect = rng.random() < self.defect_prob
        defects: List[dict] = []
        defect_type = "none"
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        if has_defect:
            defect_type = rng.choice(_DEFECT_TYPES)
            severity = float(rng.uniform(0.2, 1.0))
            material = rng.choice(_MATERIALS)

            # Random bounding box for the defect
            dh = rng.integers(self.height // 8, self.height // 3)
            dw = rng.integers(self.width // 8, self.width // 3)
            y0 = rng.integers(0, self.height - dh)
            x0 = rng.integers(0, self.width - dw)
            bbox = (int(y0), int(x0), int(y0 + dh), int(x0 + dw))

            defect_desc: dict = {
                "type": defect_type,
                "bbox": bbox,
                "severity": severity,
                "material": material,
            }
            if defect_type == "contamination":
                defect_desc["contaminant"] = rng.choice(_CONTAMINANTS)

            defects.append(defect_desc)

            # Binary mask for the defect region
            mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1.0

        # Render multi-spectral image
        image = render_spectral_image(
            layout, defects=defects, config=self.config,
            height=self.height, width=self.width, rng=rng,
        )

        return {
            "image": torch.from_numpy(image),                           # [C, H, W]
            "mask": torch.from_numpy(mask).unsqueeze(0),                # [1, H, W]
            "defect_type": str(defect_type),
        }

    def _load_from_disk(self, index: int) -> Dict[str, Any]:
        """Load a pre-generated sample from disk."""
        path = self._file_list[index]
        data = np.load(str(path), allow_pickle=True)
        image = torch.from_numpy(data["image"])
        mask = torch.from_numpy(data["mask"])
        defect_type = str(data["defect_type"])
        return {
            "image": image,
            "mask": mask,
            "defect_type": defect_type,
        }

    def __repr__(self) -> str:
        src = f"from_dir='{self.from_dir}'" if self.from_dir else "on-the-fly"
        return (
            f"SpectralDataset(n={self.n_samples}, channels={self.config.n_channels}, "
            f"size=({self.height}, {self.width}), {src})"
        )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_spectral_dataset(
    output_dir: str,
    n_samples: int = 1000,
    config: Optional[SpectralConfig] = None,
    height: int = 512,
    width: int = 512,
    defect_prob: float = 0.7,
    seed: int = 42,
) -> Path:
    """Generate a synthetic multi-spectral dataset and save to disk.

    Each sample is stored as ``sample_{index:05d}.npz`` containing keys
    ``image`` (float32 [C,H,W]), ``mask`` (float32 [1,H,W]), and
    ``defect_type`` (str).

    Parameters
    ----------
    output_dir : str
        Directory to save the dataset.
    n_samples : int
        Number of samples to generate.
    config : SpectralConfig | None
        Spectral configuration.
    height, width : int
        Image spatial dimensions.
    defect_prob : float
        Probability that a sample contains a defect.
    seed : int
        Random seed.

    Returns
    -------
    Path
        Path to the output directory.
    """
    from tqdm import tqdm

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = SpectralDataset(
        n_samples=n_samples, config=config,
        height=height, width=width,
        defect_prob=defect_prob, seed=seed,
    )

    for i in tqdm(range(n_samples), desc="Generating spectral dataset"):
        sample = ds[i]
        np.savez_compressed(
            str(out / f"sample_{i:05d}.npz"),
            image=sample["image"].numpy(),
            mask=sample["mask"].numpy(),
            defect_type=sample["defect_type"],
        )

    # Save metadata
    meta = {
        "n_samples": n_samples,
        "wavelengths": (config or default_spectral_config()).wavelengths,
        "height": height,
        "width": width,
        "defect_prob": defect_prob,
        "seed": seed,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Generated %d samples in %s", n_samples, out)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _random_layout(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random semiconductor package material layout.

    Creates a simplified top-down view with rectangular regions of
    different materials, similar to a BGA or QFP package cross-section.
    """
    layout = np.zeros((height, width), dtype=np.int32)  # default: mold_compound

    # Central die region (silicon)
    margin_h = height // 4
    margin_w = width // 4
    layout[margin_h:height - margin_h, margin_w:width - margin_w] = 1  # silicon

    # Copper traces / lead frame (random rectangular pads)
    n_pads = rng.integers(4, 12)
    for _ in range(n_pads):
        ph = rng.integers(height // 16, height // 8)
        pw = rng.integers(width // 16, width // 8)
        py = rng.integers(0, height - ph)
        px = rng.integers(0, width - pw)
        layout[py:py + ph, px:px + pw] = 2  # copper

    # Solder bumps (small squares)
    n_bumps = rng.integers(2, 8)
    for _ in range(n_bumps):
        bh = rng.integers(height // 32, height // 16)
        bw = rng.integers(width // 32, width // 16)
        by = rng.integers(0, height - bh)
        bx = rng.integers(0, width - bw)
        layout[by:by + bh, bx:bx + bw] = 3  # solder

    return layout
