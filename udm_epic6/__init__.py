"""
UDM Epic 6 — AOI Bond Wire & Surface Defect Synthesis.

Generate synthetic Automatic Optical Inspection (AOI) images with
bent, broken, and lifted bond wires for semiconductor defect detection
training.

Install extras: ``pip install -e ".[epic6]"``
"""

from udm_epic6.models.wire_geometry import BondWireProfile, generate_wire_profile, render_wire_mask
from udm_epic6.models.defect_generator import apply_bend_defect, apply_break_defect, apply_lift_defect
from udm_epic6.rendering.aoi_renderer import render_aoi_image, render_background
from udm_epic6.data.dataset import BondWireDataset, generate_bond_wire_dataset

__all__ = [
    "BondWireProfile",
    "generate_wire_profile",
    "render_wire_mask",
    "apply_bend_defect",
    "apply_break_defect",
    "apply_lift_defect",
    "render_aoi_image",
    "render_background",
    "BondWireDataset",
    "generate_bond_wire_dataset",
]
