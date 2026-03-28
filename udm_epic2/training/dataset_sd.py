"""PyTorch dataset for ControlNet fine-tuning from Epic 2 crop exports."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Epic2ControlNetDataset(Dataset):
    """
    Reads ``manifest.csv`` under ``crops_root`` with ``image_relpath`` and ``edge_relpath``.
    """

    def __init__(
        self,
        crops_root: Path,
        resolution: int = 512,
        caption: str = "semiconductor x-ray void defect",
        tokenizer: Optional[object] = None,
        manifest_name: str = "manifest.csv",
    ):
        self.crops_root = Path(crops_root)
        self.resolution = resolution
        self.caption = caption
        self.tokenizer = tokenizer

        df = pd.read_csv(self.crops_root / manifest_name)
        if "image_relpath" not in df.columns or "edge_relpath" not in df.columns:
            raise ValueError("manifest must include image_relpath and edge_relpath")
        self.rows = df.to_dict("records")

        self.image_tfm = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.cond_tfm = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        img_path = self.crops_root / str(r["image_relpath"])
        edge_path = self.crops_root / str(r["edge_relpath"])
        image = Image.open(img_path).convert("RGB")
        edge = Image.open(edge_path).convert("RGB")

        pixel_values = self.image_tfm(image)
        # conditioning: 3-channel (edge map repeated)
        conditioning = self.cond_tfm(edge)

        item = {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning,
        }
        if self.tokenizer is not None:
            ids = self.tokenizer(
                self.caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            item["input_ids"] = ids
        return item


def collate_fn(batch: list) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).float()
    cond = torch.stack([b["conditioning_pixel_values"] for b in batch]).float()
    out = {"pixel_values": pixel_values, "conditioning_pixel_values": cond}
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    return out
