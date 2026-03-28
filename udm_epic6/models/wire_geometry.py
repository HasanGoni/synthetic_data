"""
Epic 6 — Bond wire geometry: profile dataclass, random generation, and mask rendering.

Bond wires in semiconductor packages follow a loop shape from one bond pad
to another.  We approximate the wire path with a cubic Bezier curve whose
control points encode start, end, loop height, and curvature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import Generator


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class BondWireProfile:
    """Geometric description of a single bond wire.

    Attributes:
        start_xy:   (x, y) bond pad origin in pixel coordinates.
        end_xy:     (x, y) bond pad destination in pixel coordinates.
        loop_height: Peak height of the wire loop (pixels above the baseline).
        diameter:    Wire diameter in pixels.
        curvature:   Asymmetry factor in [-1, 1].  0 = symmetric loop,
                     positive = peak shifted toward start, negative = toward end.
        material:    Wire material — affects reflectance colour.
    """

    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    loop_height: float = 40.0
    diameter: float = 3.0
    curvature: float = 0.0
    material: str = "gold"  # gold | copper | aluminum


# ── Random generation ─────────────────────────────────────────────────────────


def generate_wire_profile(
    rng: Generator,
    image_size: Tuple[int, int] = (512, 512),
    n_wires: Optional[int] = None,
) -> List[BondWireProfile]:
    """Generate random but plausible bond wire profiles.

    Wire endpoints are placed in the pad regions (left/right margins) and the
    loop height is proportional to the span length, following typical
    semiconductor packaging rules.

    Args:
        rng:        NumPy random Generator for reproducibility.
        image_size: (height, width) of the target image.
        n_wires:    Number of wires to generate.  If *None*, a random count
                    in [1, 5] is chosen.

    Returns:
        List of :class:`BondWireProfile` instances.
    """
    h, w = image_size
    if n_wires is None:
        n_wires = int(rng.integers(1, 6))

    materials = ["gold", "copper", "aluminum"]
    profiles: List[BondWireProfile] = []

    # Distribute wires evenly across vertical extent with jitter
    y_positions = np.linspace(h * 0.15, h * 0.85, n_wires)

    for i in range(n_wires):
        # Start pad: left side of die
        sx = float(rng.uniform(w * 0.05, w * 0.20))
        sy = float(y_positions[i] + rng.normal(0, h * 0.02))
        sy = float(np.clip(sy, h * 0.05, h * 0.95))

        # End pad: right side (lead frame)
        ex = float(rng.uniform(w * 0.80, w * 0.95))
        ey = float(sy + rng.normal(0, h * 0.05))
        ey = float(np.clip(ey, h * 0.05, h * 0.95))

        span = np.hypot(ex - sx, ey - sy)
        loop_height = float(rng.uniform(0.10, 0.25) * span)
        diameter = float(rng.uniform(1.5, 4.0))
        curvature = float(rng.uniform(-0.3, 0.3))
        material = materials[int(rng.integers(0, len(materials)))]

        profiles.append(
            BondWireProfile(
                start_xy=(sx, sy),
                end_xy=(ex, ey),
                loop_height=loop_height,
                diameter=diameter,
                curvature=curvature,
                material=material,
            )
        )

    return profiles


# ── Bezier helpers ────────────────────────────────────────────────────────────


def _bezier_control_points(profile: BondWireProfile) -> np.ndarray:
    """Compute cubic Bezier control points for a wire loop.

    The wire runs from *start_xy* to *end_xy* with a loop whose peak
    is *loop_height* pixels above the baseline connecting the endpoints.
    The *curvature* parameter shifts the peak toward start (>0) or end (<0).

    Returns:
        Array of shape ``(4, 2)`` — four control points ``[P0, P1, P2, P3]``.
    """
    p0 = np.array(profile.start_xy, dtype=np.float64)
    p3 = np.array(profile.end_xy, dtype=np.float64)

    mid = (p0 + p3) / 2.0
    direction = p3 - p0
    # Normal to the wire direction (pointing "up" for a left-to-right wire)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-6:
        normal /= norm_len

    # Shift control points along the direction for curvature asymmetry
    shift = profile.curvature * 0.25 * direction

    p1 = mid - shift - normal * profile.loop_height * 0.9
    p2 = mid + shift - normal * profile.loop_height * 0.9

    return np.stack([p0, p1, p2, p3])


def _evaluate_bezier(control_points: np.ndarray, n_samples: int = 200) -> np.ndarray:
    """Evaluate cubic Bezier curve at *n_samples* uniformly spaced *t* values.

    Args:
        control_points: Shape ``(4, 2)`` — P0, P1, P2, P3.
        n_samples:      Number of points along the curve.

    Returns:
        Array of shape ``(n_samples, 2)`` — (x, y) curve coordinates.
    """
    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    p0, p1, p2, p3 = control_points

    # De Casteljau / explicit cubic Bezier
    curve = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )
    return curve


# ── Mask rendering ────────────────────────────────────────────────────────────


def render_wire_mask(
    profile: BondWireProfile,
    height: int,
    width: int,
    n_curve_samples: int = 300,
) -> np.ndarray:
    """Render a binary mask of a single bond wire.

    The wire path is a cubic Bezier curve; the mask is produced by dilating
    the path to the wire diameter using distance-from-curve thresholding.

    Args:
        profile:          Wire profile describing geometry and material.
        height:           Image height in pixels.
        width:            Image width in pixels.
        n_curve_samples:  Number of points sampled along the Bezier curve.

    Returns:
        Binary ``uint8`` mask of shape ``(height, width)`` where 255 marks
        the wire region.
    """
    cps = _bezier_control_points(profile)
    curve = _evaluate_bezier(cps, n_samples=n_curve_samples)

    # Build pixel grid
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    coords = np.stack([xx, yy], axis=-1)  # (H, W, 2)

    # Compute min distance from every pixel to the curve
    # For efficiency, broadcast: coords (H,W,1,2) vs curve (1,1,N,2)
    diff = coords[:, :, np.newaxis, :] - curve[np.newaxis, np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=-1)  # (H, W)

    radius = profile.diameter / 2.0
    mask = (dist <= radius).astype(np.uint8) * 255
    return mask
