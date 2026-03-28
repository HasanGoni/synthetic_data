"""
Epic 9 — Physics-based crack pattern generation.

Generates realistic crack paths using fractal midpoint displacement and
recursive branching, then renders them as binary masks with variable-width
lines suitable for semiconductor defect synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class CrackProfile:
    """Specification for a single crack pattern.

    Parameters
    ----------
    start_xy : tuple[float, float]
        Starting point ``(x, y)`` in pixel coordinates.
    end_xy : tuple[float, float]
        Ending point ``(x, y)`` in pixel coordinates.
    width_range : tuple[int, int]
        Min and max width of the crack line in pixels.
    branching_prob : float
        Probability of spawning a branch at each recursive step.
    roughness : float
        Controls fractal displacement amplitude (0 = straight, 1 = very rough).
    crack_type : str
        One of ``'linear'``, ``'branching'``, ``'network'``, ``'radial'``.
    """

    start_xy: Tuple[float, float] = (0.0, 0.0)
    end_xy: Tuple[float, float] = (1.0, 1.0)
    width_range: Tuple[int, int] = (1, 4)
    branching_prob: float = 0.3
    roughness: float = 0.3
    crack_type: str = "linear"


# ── Path generation ────────────────────────────────────────────────────────


def generate_crack_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    roughness: float = 0.3,
    n_points: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a rough crack path using random midpoint displacement.

    The algorithm recursively subdivides a straight line between *start* and
    *end*, displacing each midpoint perpendicular to the segment direction by
    an amount proportional to *roughness* and the segment length.  The result
    is a fractal polyline that mimics natural crack propagation.

    Parameters
    ----------
    start : tuple[float, float]
        Starting ``(x, y)`` coordinate.
    end : tuple[float, float]
        Ending ``(x, y)`` coordinate.
    roughness : float
        Displacement scale relative to segment length (0..1).
    n_points : int
        Approximate number of output vertices (determines recursion depth).
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with ``(x, y)`` coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine recursion depth from desired point count
    depth = max(1, int(np.log2(max(n_points, 2))))

    points = [np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)]

    for _ in range(depth):
        new_points = [points[0]]
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            mid = 0.5 * (p0 + p1)
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            # Perpendicular direction
            if seg_len > 0:
                perp = np.array([-seg[1], seg[0]]) / seg_len
            else:
                perp = np.array([0.0, 1.0])
            displacement = roughness * seg_len * rng.normal(0, 0.5)
            mid = mid + perp * displacement
            new_points.extend([mid, p1])
        points = new_points

    return np.array(points, dtype=np.float64)


def generate_branching_crack(
    origin: Tuple[float, float],
    n_branches: Optional[int] = None,
    max_depth: int = 3,
    rng: Optional[np.random.Generator] = None,
    *,
    height: int = 512,
    width: int = 512,
    roughness: float = 0.3,
    _depth: int = 0,
) -> List[np.ndarray]:
    """Generate a recursive branching crack tree.

    Starting from *origin*, generates a main crack path and recursively
    spawns branches at random points along the path, creating a
    tree-like crack structure typical of brittle fracture.

    Parameters
    ----------
    origin : tuple[float, float]
        Root ``(x, y)`` of the crack tree.
    n_branches : int, optional
        Number of branches from the root (random in 2..5 if ``None``).
    max_depth : int
        Maximum recursion depth for sub-branches.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.
    height, width : int
        Image dimensions for bounding end-points.
    roughness : float
        Crack roughness parameter.

    Returns
    -------
    list[np.ndarray]
        List of crack paths, each of shape ``(N, 2)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if _depth >= max_depth:
        return []

    if n_branches is None:
        n_branches = rng.integers(2, 6)

    paths: List[np.ndarray] = []
    for _ in range(n_branches):
        # Generate random end point, biased away from origin
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.15, 0.5) * min(height, width)
        # Reduce length at deeper recursion
        length *= 0.6 ** _depth
        end_x = origin[0] + length * np.cos(angle)
        end_y = origin[1] + length * np.sin(angle)
        # Clamp to image bounds
        end_x = float(np.clip(end_x, 0, width - 1))
        end_y = float(np.clip(end_y, 0, height - 1))
        end = (end_x, end_y)

        path = generate_crack_path(origin, end, roughness=roughness, n_points=30, rng=rng)
        paths.append(path)

        # Recursively branch from a random point along this path
        if _depth < max_depth - 1 and len(path) > 2:
            branch_idx = rng.integers(len(path) // 3, len(path))
            branch_origin = tuple(path[branch_idx])
            sub_branches = generate_branching_crack(
                origin=branch_origin,
                n_branches=rng.integers(1, 3),
                max_depth=max_depth,
                rng=rng,
                height=height,
                width=width,
                roughness=roughness,
                _depth=_depth + 1,
            )
            paths.extend(sub_branches)

    return paths


def generate_crack_network(
    height: int,
    width: int,
    n_cracks: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    roughness: float = 0.3,
) -> List[np.ndarray]:
    """Generate multiple intersecting cracks forming a network pattern.

    Produces a set of independent crack paths with random start/end points
    across the image, simulating stress-induced crack networks in mold
    compound or substrate layers.

    Parameters
    ----------
    height, width : int
        Image dimensions.
    n_cracks : int, optional
        Number of cracks (random in 3..8 if ``None``).
    rng : numpy.random.Generator, optional
        Random generator.
    roughness : float
        Crack roughness parameter.

    Returns
    -------
    list[np.ndarray]
        List of crack paths, each of shape ``(N, 2)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_cracks is None:
        n_cracks = rng.integers(3, 9)

    paths: List[np.ndarray] = []
    for _ in range(n_cracks):
        start = (rng.uniform(0, width), rng.uniform(0, height))
        end = (rng.uniform(0, width), rng.uniform(0, height))
        path = generate_crack_path(start, end, roughness=roughness, n_points=50, rng=rng)
        paths.append(path)

    return paths


# ── Mask rendering ─────────────────────────────────────────────────────────


def render_crack_mask(
    crack_paths: List[np.ndarray],
    height: int,
    width: int,
    width_range: Tuple[int, int] = (1, 4),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render a binary crack mask from a list of polyline paths.

    Each path is drawn as a polyline with variable width sampled uniformly
    from *width_range*.  The result is a binary ``uint8`` mask where crack
    pixels are 255 and background is 0.

    Parameters
    ----------
    crack_paths : list[np.ndarray]
        Polyline paths, each of shape ``(N, 2)`` with ``(x, y)`` coords.
    height, width : int
        Output mask dimensions.
    width_range : tuple[int, int]
        Min and max line thickness in pixels.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(height, width)`` with dtype ``uint8``.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros((height, width), dtype=np.uint8)

    for path in crack_paths:
        if len(path) < 2:
            continue
        # Convert to integer pixel coords and clip to bounds
        pts = np.clip(path, [0, 0], [width - 1, height - 1]).astype(np.int32)
        # Sample a line thickness for this crack segment
        thickness = int(rng.integers(width_range[0], width_range[1] + 1))
        # Draw polyline
        pts_cv = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts_cv], isClosed=False, color=255, thickness=thickness)

    return mask
