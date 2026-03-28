"""
UDM Epic 2 — ControlNet data prep, training, generation, paste (Notion Epic 2).

Install training/generation extras: ``pip install -e ".[epic2]"``.
"""

from udm_epic2.conditioning.edges import edge_map_from_mask
from udm_epic2.dataset.crops import CropConfig, extract_crops_for_pair, process_crop_dataset

__all__ = [
    "CropConfig",
    "extract_crops_for_pair",
    "process_crop_dataset",
    "edge_map_from_mask",
]
