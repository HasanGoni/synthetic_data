"""
Epic 9 — Semiconductor-specific crack type generators.

Each function produces a binary crack mask and metadata dict describing
the generated pattern.  The four crack types correspond to common failure
modes observed in semiconductor packaging:

- **Die crack**: radial cracks from corners/edges of the silicon die
- **Substrate crack**: linear cracks following PCB stress lines
- **Mold crack**: network-pattern cracks in mold compound
- **Delamination crack**: interface-following cracks at layer boundaries
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from udm_epic9.models.crack_geometry import (
    generate_branching_crack,
    generate_crack_network,
    generate_crack_path,
    render_crack_mask,
)

logger = logging.getLogger(__name__)


def die_crack(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate a die crack pattern — radial cracks from corners or edges.

    Silicon die cracks typically originate at corners or edge defects and
    propagate radially inward.  This function picks 1-3 corner/edge origins
    and generates branching cracks from each.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    tuple[np.ndarray, dict]
        Binary mask ``(height, width)`` uint8, and metadata dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Possible origins: corners and edge midpoints
    corners = [
        (0.0, 0.0), (float(width - 1), 0.0),
        (0.0, float(height - 1)), (float(width - 1), float(height - 1)),
    ]
    edges = [
        (float(width // 2), 0.0), (float(width // 2), float(height - 1)),
        (0.0, float(height // 2)), (float(width - 1), float(height // 2)),
    ]
    candidates = corners + edges

    n_origins = rng.integers(1, 4)
    selected = rng.choice(len(candidates), size=n_origins, replace=False)

    all_paths: List[np.ndarray] = []
    origins_used = []
    for idx in selected:
        origin = candidates[idx]
        origins_used.append(origin)
        paths = generate_branching_crack(
            origin=origin,
            n_branches=rng.integers(2, 5),
            max_depth=3,
            rng=rng,
            height=height,
            width=width,
            roughness=rng.uniform(0.15, 0.35),
        )
        all_paths.extend(paths)

    mask = render_crack_mask(all_paths, height, width, width_range=(1, 3), rng=rng)
    metadata = {
        "crack_type": "die_crack",
        "n_origins": int(n_origins),
        "origins": origins_used,
        "n_paths": len(all_paths),
    }
    return mask, metadata


def substrate_crack(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate a substrate crack — linear cracks following stress lines.

    Substrate cracks tend to be predominantly horizontal or vertical,
    following the grain of the PCB laminate.  This function generates
    1-3 roughly aligned cracks with low roughness.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple[np.ndarray, dict]
        Binary mask and metadata dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cracks = rng.integers(1, 4)
    # Dominant direction: 0 = horizontal, 1 = vertical
    direction = int(rng.integers(0, 2))
    all_paths: List[np.ndarray] = []

    for _ in range(n_cracks):
        if direction == 0:  # horizontal
            y = rng.uniform(height * 0.2, height * 0.8)
            start = (0.0, float(y + rng.normal(0, 5)))
            end = (float(width - 1), float(y + rng.normal(0, 10)))
        else:  # vertical
            x = rng.uniform(width * 0.2, width * 0.8)
            start = (float(x + rng.normal(0, 5)), 0.0)
            end = (float(x + rng.normal(0, 10)), float(height - 1))

        path = generate_crack_path(start, end, roughness=rng.uniform(0.1, 0.25), n_points=60, rng=rng)
        all_paths.append(path)

    mask = render_crack_mask(all_paths, height, width, width_range=(1, 3), rng=rng)
    metadata = {
        "crack_type": "substrate_crack",
        "n_cracks": int(n_cracks),
        "direction": "horizontal" if direction == 0 else "vertical",
        "n_paths": len(all_paths),
    }
    return mask, metadata


def mold_crack(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate a mold compound crack — network pattern of intersecting cracks.

    Mold compound cracks form complex networks due to thermal stress and
    moisture absorption.  Uses :func:`generate_crack_network` with moderate
    to high crack count and roughness.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple[np.ndarray, dict]
        Binary mask and metadata dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cracks = rng.integers(4, 9)
    roughness = rng.uniform(0.2, 0.5)
    paths = generate_crack_network(
        height, width, n_cracks=n_cracks, rng=rng, roughness=roughness,
    )

    mask = render_crack_mask(paths, height, width, width_range=(1, 4), rng=rng)
    metadata = {
        "crack_type": "mold_crack",
        "n_cracks": int(n_cracks),
        "roughness": float(roughness),
        "n_paths": len(paths),
    }
    return mask, metadata


def delamination_crack(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """Generate a delamination-induced crack following an interface boundary.

    Delamination cracks propagate along the interface between two layers
    (e.g., die-attach to substrate).  They tend to be long, roughly
    horizontal, and follow a gently curving boundary with occasional
    short branches.

    Parameters
    ----------
    height, width : int
        Output image dimensions.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    tuple[np.ndarray, dict]
        Binary mask and metadata dict.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Interface is a horizontal band somewhere in the image
    interface_y = rng.uniform(height * 0.3, height * 0.7)
    all_paths: List[np.ndarray] = []

    # Main delamination crack — long, follows interface
    start = (rng.uniform(0, width * 0.15), float(interface_y + rng.normal(0, 3)))
    end = (rng.uniform(width * 0.85, width), float(interface_y + rng.normal(0, 3)))
    main_path = generate_crack_path(start, end, roughness=rng.uniform(0.08, 0.2), n_points=80, rng=rng)
    all_paths.append(main_path)

    # Short branches off the main crack
    n_branches = rng.integers(0, 4)
    for _ in range(n_branches):
        branch_idx = rng.integers(len(main_path) // 4, 3 * len(main_path) // 4)
        branch_origin = tuple(main_path[branch_idx])
        angle = rng.uniform(-np.pi / 3, np.pi / 3) + (np.pi / 2 if rng.random() > 0.5 else -np.pi / 2)
        length = rng.uniform(15, 60)
        end_x = float(np.clip(branch_origin[0] + length * np.cos(angle), 0, width - 1))
        end_y = float(np.clip(branch_origin[1] + length * np.sin(angle), 0, height - 1))
        branch_path = generate_crack_path(
            branch_origin, (end_x, end_y), roughness=rng.uniform(0.15, 0.3), n_points=20, rng=rng,
        )
        all_paths.append(branch_path)

    mask = render_crack_mask(all_paths, height, width, width_range=(1, 3), rng=rng)
    metadata = {
        "crack_type": "delamination_crack",
        "interface_y": float(interface_y),
        "n_branches": int(n_branches),
        "n_paths": len(all_paths),
    }
    return mask, metadata
