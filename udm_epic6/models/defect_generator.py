"""
Epic 6 — Defect generators for bond wire anomalies.

Three defect types common in AOI inspection:

* **Bend**: wire deformed laterally or vertically (sagging / sweeping).
* **Break**: wire fractured at some point along its length.
* **Lift**: wire bond pad detachment (lift-off), raising one end.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from udm_epic6.models.wire_geometry import BondWireProfile


def _ensure_rng(rng: Optional[Generator]) -> Generator:
    """Return *rng* if given, else a fresh default Generator."""
    if rng is None:
        return np.random.default_rng()
    return rng


def apply_bend_defect(
    wire_profile: BondWireProfile,
    severity: float = 0.5,
    rng: Optional[Generator] = None,
) -> BondWireProfile:
    """Simulate a bent (swept) wire by deforming the loop geometry.

    A bent wire has its midpoint displaced laterally and/or vertically.
    Severity controls the magnitude of the displacement:

    * 0.1 — slight sag, barely visible
    * 0.5 — clear wire sweep
    * 0.9 — extreme deformation, near-short risk

    Args:
        wire_profile: Original wire profile to deform.
        severity:     Defect magnitude in [0.1, 0.9].
        rng:          NumPy Generator for reproducibility.

    Returns:
        New :class:`BondWireProfile` with modified geometry.
    """
    rng = _ensure_rng(rng)
    severity = float(np.clip(severity, 0.1, 0.9))

    profile = copy.deepcopy(wire_profile)

    sx, sy = profile.start_xy
    ex, ey = profile.end_xy
    span = np.hypot(ex - sx, ey - sy)

    # Lateral displacement perpendicular to wire direction
    lateral_shift = severity * span * 0.15 * rng.choice([-1.0, 1.0])

    direction = np.array([ex - sx, ey - sy], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        normal = np.array([-direction[1], direction[0]]) / norm
    else:
        normal = np.array([0.0, 1.0])

    # Apply lateral shift to the midpoint via curvature adjustment
    profile.curvature = float(profile.curvature + severity * rng.uniform(-0.6, 0.6))
    profile.curvature = float(np.clip(profile.curvature, -0.9, 0.9))

    # Modify loop height — bent wires often sag or bow
    height_factor = 1.0 + severity * rng.uniform(-0.5, 0.8)
    profile.loop_height = float(max(5.0, profile.loop_height * height_factor))

    # Slightly shift endpoints to simulate mechanical stress
    shift_amount = severity * span * 0.03
    profile.start_xy = (
        sx + float(rng.normal(0, shift_amount)),
        sy + float(rng.normal(0, shift_amount)),
    )
    profile.end_xy = (
        ex + float(rng.normal(0, shift_amount)),
        ey + float(rng.normal(0, shift_amount)),
    )

    return profile


def apply_break_defect(
    wire_profile: BondWireProfile,
    break_position: float = 0.5,
    rng: Optional[Generator] = None,
) -> Tuple[BondWireProfile, BondWireProfile]:
    """Simulate a broken wire by splitting at *break_position*.

    The wire is divided into two segments: one retracting toward start,
    the other toward end.  Each segment curls slightly, mimicking real
    fractured bond wires under residual stress.

    Args:
        wire_profile:   Original wire profile.
        break_position: Normalised position along the wire in [0.0, 1.0]
                        where the break occurs (0 = near start, 1 = near end).
        rng:            NumPy Generator for reproducibility.

    Returns:
        Tuple of two :class:`BondWireProfile` — the start-side and end-side
        wire fragments.
    """
    rng = _ensure_rng(rng)
    t = float(np.clip(break_position, 0.05, 0.95))

    sx, sy = wire_profile.start_xy
    ex, ey = wire_profile.end_xy

    # Compute the break point via linear interpolation on the baseline
    bx = sx + t * (ex - sx)
    by = sy + t * (ey - sy)

    # Retraction: each fragment pulls back slightly from the break point
    retract = wire_profile.diameter * rng.uniform(1.0, 3.0)
    direction = np.array([ex - sx, ey - sy], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        unit = direction / norm
    else:
        unit = np.array([1.0, 0.0])

    # Fragment 1: start -> break (pulled back toward start)
    frag1_end = (
        float(bx - retract * unit[0]),
        float(by - retract * unit[1]),
    )
    frag1 = BondWireProfile(
        start_xy=wire_profile.start_xy,
        end_xy=frag1_end,
        loop_height=wire_profile.loop_height * t * rng.uniform(0.6, 1.0),
        diameter=wire_profile.diameter,
        curvature=float(rng.uniform(-0.2, 0.2)),
        material=wire_profile.material,
    )

    # Fragment 2: break -> end (pulled back toward end)
    frag2_start = (
        float(bx + retract * unit[0]),
        float(by + retract * unit[1]),
    )
    frag2 = BondWireProfile(
        start_xy=frag2_start,
        end_xy=wire_profile.end_xy,
        loop_height=wire_profile.loop_height * (1 - t) * rng.uniform(0.6, 1.0),
        diameter=wire_profile.diameter,
        curvature=float(rng.uniform(-0.2, 0.2)),
        material=wire_profile.material,
    )

    return frag1, frag2


def apply_lift_defect(
    wire_profile: BondWireProfile,
    lift_amount: Optional[float] = None,
    rng: Optional[Generator] = None,
) -> BondWireProfile:
    """Simulate wire lift-off from the bond pad.

    In a lift defect the wire detaches from one pad, causing the start or
    end to rise.  This increases the loop height on one side and shifts the
    affected endpoint upward.

    Args:
        wire_profile: Original wire profile.
        lift_amount:  Pixels of vertical lift.  If *None*, a random amount
                      proportional to the loop height is chosen.
        rng:          NumPy Generator for reproducibility.

    Returns:
        New :class:`BondWireProfile` with lifted endpoint.
    """
    rng = _ensure_rng(rng)
    profile = copy.deepcopy(wire_profile)

    if lift_amount is None:
        lift_amount = float(rng.uniform(0.3, 0.8) * profile.loop_height)

    # Choose which end lifts
    lift_start = bool(rng.integers(0, 2))

    if lift_start:
        sx, sy = profile.start_xy
        profile.start_xy = (sx, sy - lift_amount)
        # Increase loop height asymmetry
        profile.curvature = float(np.clip(profile.curvature + 0.4, -0.9, 0.9))
    else:
        ex, ey = profile.end_xy
        profile.end_xy = (ex, ey - lift_amount)
        profile.curvature = float(np.clip(profile.curvature - 0.4, -0.9, 0.9))

    # A lifted wire often has increased overall loop height
    profile.loop_height = float(profile.loop_height + lift_amount * 0.5)

    return profile
