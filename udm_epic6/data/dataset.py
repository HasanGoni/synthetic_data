"""
Epic 6 — On-the-fly bond wire dataset and offline generation utility.

:class:`BondWireDataset` is a PyTorch :class:`~torch.utils.data.Dataset`
that synthesises images at ``__getitem__`` time — no disk I/O, fully
deterministic given a seed.

:func:`generate_bond_wire_dataset` writes images, masks, and metadata to
disk for offline training pipelines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from udm_epic6.models.wire_geometry import (
    BondWireProfile,
    generate_wire_profile,
    render_wire_mask,
)
from udm_epic6.models.defect_generator import (
    apply_bend_defect,
    apply_break_defect,
    apply_lift_defect,
)
from udm_epic6.rendering.aoi_renderer import render_aoi_image

logger = logging.getLogger(__name__)

# Default defect probabilities when no config is provided
_DEFAULT_DEFECT_PROBS: Dict[str, float] = {
    "none": 0.30,
    "bend": 0.25,
    "break": 0.25,
    "lift": 0.20,
}


def _pick_defect(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    """Randomly choose a defect type according to *probs*."""
    types = list(probs.keys())
    weights = np.array([probs[t] for t in types], dtype=np.float64)
    weights /= weights.sum()
    return types[int(rng.choice(len(types), p=weights))]


class BondWireDataset(Dataset):
    """On-the-fly synthetic bond wire AOI dataset.

    Each call to ``__getitem__`` deterministically generates:

    * An RGB AOI image as a ``float32`` tensor ``[3, H, W]`` in [0, 1].
    * A binary defect mask tensor ``[1, H, W]``.
    * A defect type string.
    * Metadata dict with wire count, material, defect severity, etc.

    Args:
        n_samples:    Virtual dataset size.
        image_size:   ``(height, width)`` tuple.
        seed:         Base seed for reproducible generation.
        defect_probs: Mapping ``{defect_type: probability}``.
        wire_range:   ``(min_wires, max_wires)`` per image.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: tuple[int, int] = (512, 512),
        seed: int = 42,
        defect_probs: Optional[Dict[str, float]] = None,
        wire_range: tuple[int, int] = (1, 5),
    ) -> None:
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self.defect_probs = defect_probs or _DEFAULT_DEFECT_PROBS
        self.wire_range = wire_range

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + idx)
        h, w = self.image_size

        n_wires = int(rng.integers(self.wire_range[0], self.wire_range[1] + 1))
        profiles = generate_wire_profile(rng, image_size=(h, w), n_wires=n_wires)

        # Choose defect type for this sample
        defect_type = _pick_defect(rng, self.defect_probs)

        defect_labels: List[Optional[str]] = [None] * len(profiles)
        defect_profiles = list(profiles)
        severity = 0.0

        if defect_type != "none" and len(profiles) > 0:
            # Apply defect to a randomly chosen wire
            target_idx = int(rng.integers(0, len(profiles)))
            severity = float(rng.uniform(0.2, 0.8))

            if defect_type == "bend":
                defect_profiles[target_idx] = apply_bend_defect(
                    profiles[target_idx], severity=severity, rng=rng,
                )
                defect_labels[target_idx] = "bend"

            elif defect_type == "break":
                frag1, frag2 = apply_break_defect(
                    profiles[target_idx],
                    break_position=float(rng.uniform(0.2, 0.8)),
                    rng=rng,
                )
                # Replace the original wire with its two fragments
                defect_profiles[target_idx] = frag1
                defect_profiles.append(frag2)
                defect_labels[target_idx] = "break"
                defect_labels.append("break")

            elif defect_type == "lift":
                defect_profiles[target_idx] = apply_lift_defect(
                    profiles[target_idx], rng=rng,
                )
                defect_labels[target_idx] = "lift"

        # Render image
        image = render_aoi_image(defect_profiles, defect_labels, h, w, rng)

        # Build combined defect mask (union of all defective wire masks)
        mask = np.zeros((h, w), dtype=np.uint8)
        for prof, label in zip(defect_profiles, defect_labels):
            if label is not None:
                wire_mask = render_wire_mask(prof, h, w)
                mask = np.maximum(mask, wire_mask)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        metadata = {
            "n_wires": n_wires,
            "defect_type": defect_type,
            "severity": severity,
            "materials": [p.material for p in defect_profiles],
            "idx": idx,
        }

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "defect_type": defect_type,
            "metadata": metadata,
        }


def generate_bond_wire_dataset(
    output_dir: str | Path,
    n_samples: int = 1000,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a bond wire dataset and save to disk.

    Creates the following directory structure::

        output_dir/
            images/      — PNG images
            masks/       — PNG binary masks
            metadata.json — per-sample metadata

    Args:
        output_dir: Root directory to write into.
        n_samples:  Number of samples to generate.
        config:     Optional config dict overriding defaults (image_size,
                    defect_probs, wire_range, seed).

    Returns:
        Path to the *output_dir*.
    """
    import cv2

    config = config or {}
    out = Path(output_dir)
    img_dir = out / "images"
    mask_dir = out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_size = tuple(config.get("image_size", (512, 512)))
    seed = config.get("seed", 42)
    defect_probs = config.get("defect_probs", None)
    wire_range = tuple(config.get("wire_range", (1, 5)))

    ds = BondWireDataset(
        n_samples=n_samples,
        image_size=image_size,
        seed=seed,
        defect_probs=defect_probs,
        wire_range=wire_range,
    )

    all_metadata: List[Dict[str, Any]] = []

    for i in range(n_samples):
        sample = ds[i]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = (sample["mask"].squeeze(0).numpy() * 255).astype(np.uint8)

        cv2.imwrite(str(img_dir / f"{i:06d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{i:06d}.png"), mask_np)
        all_metadata.append(sample["metadata"])

        if (i + 1) % 100 == 0:
            logger.info("Generated %d / %d samples", i + 1, n_samples)

    with open(out / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)

    logger.info("Dataset saved to %s  (%d samples)", out, n_samples)
    return out
