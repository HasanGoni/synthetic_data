"""
Epic 6 — AOI image renderer.

Composes a synthetic Automatic Optical Inspection (AOI) image from:

1. **Background**: die surface with bond pads and traces.
2. **Wires**: bond wires rendered with specular reflection.
3. **Defects**: optional visual artifacts on defective wires.

The optical model is intentionally simple — specular highlights on wires
(approximated as cylinders) and diffuse shading on the substrate.  This
produces images that are visually plausible for training without requiring
a full ray tracer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from udm_epic6.models.wire_geometry import BondWireProfile, render_wire_mask, _bezier_control_points, _evaluate_bezier


# ── Material reflectance colours (BGR-ish, 0-255 scale) ──────────────────────

_MATERIAL_COLOUR: Dict[str, np.ndarray] = {
    "gold": np.array([210, 190, 90], dtype=np.float64),
    "copper": np.array([180, 120, 70], dtype=np.float64),
    "aluminum": np.array([200, 200, 200], dtype=np.float64),
}


def _ensure_rng(rng: Optional[Generator]) -> Generator:
    if rng is None:
        return np.random.default_rng()
    return rng


# ── Background ────────────────────────────────────────────────────────────────


def render_background(
    height: int = 512,
    width: int = 512,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Render a synthetic die surface with bond pads and traces.

    The background simulates a top-down AOI view of a semiconductor die:

    * Dark silicon substrate with slight texture noise.
    * Metallic bond pads along left and right edges.
    * Thin metallic traces connecting pads to circuit area.

    Args:
        height: Image height in pixels.
        width:  Image width in pixels.
        rng:    NumPy Generator for reproducibility.

    Returns:
        ``uint8`` RGB image of shape ``(height, width, 3)``.
    """
    rng = _ensure_rng(rng)

    # Base substrate — dark grey with Gaussian noise
    base_intensity = rng.uniform(30, 60)
    bg = rng.normal(base_intensity, 5, size=(height, width, 3)).astype(np.float64)

    # Add subtle low-frequency texture (die surface grain)
    grain = _low_freq_noise(height, width, rng, scale=16)
    bg += grain[:, :, np.newaxis] * 8.0

    # ── Bond pads ──
    pad_colour = np.array([180, 170, 100], dtype=np.float64)  # gold-ish
    n_pads = rng.integers(3, 8)
    pad_h = int(height * 0.04)
    pad_w = int(width * 0.06)

    y_positions = np.linspace(height * 0.12, height * 0.88, n_pads).astype(int)

    for yp in y_positions:
        # Left pad
        y0 = max(0, yp - pad_h // 2)
        y1 = min(height, yp + pad_h // 2)
        lx0 = int(width * 0.06)
        lx1 = lx0 + pad_w
        bg[y0:y1, lx0:lx1] = pad_colour + rng.normal(0, 3, size=(y1 - y0, lx1 - lx0, 3))

        # Right pad
        rx1 = int(width * 0.94)
        rx0 = rx1 - pad_w
        bg[y0:y1, rx0:rx1] = pad_colour + rng.normal(0, 3, size=(y1 - y0, rx1 - rx0, 3))

        # Thin trace from left pad into die centre
        trace_y0 = max(0, yp - 1)
        trace_y1 = min(height, yp + 2)
        trace_x1 = int(width * 0.40 + rng.uniform(-width * 0.05, width * 0.05))
        bg[trace_y0:trace_y1, lx1:trace_x1] = (
            pad_colour * 0.6
            + rng.normal(0, 2, size=(trace_y1 - trace_y0, max(0, trace_x1 - lx1), 3))
        )

    # ── Die outline (faint border) ──
    border_w = 2
    border_colour = np.array([100, 100, 80], dtype=np.float64)
    bg[:border_w, :] = border_colour
    bg[-border_w:, :] = border_colour
    bg[:, :border_w] = border_colour
    bg[:, -border_w:] = border_colour

    return np.clip(bg, 0, 255).astype(np.uint8)


def _low_freq_noise(height: int, width: int, rng: Generator, scale: int = 16) -> np.ndarray:
    """Generate low-frequency spatial noise via upsampled random field."""
    small_h = max(2, height // scale)
    small_w = max(2, width // scale)
    small = rng.standard_normal((small_h, small_w)).astype(np.float64)

    # Bilinear upscale using simple interpolation (no OpenCV dependency here)
    row_idx = np.linspace(0, small_h - 1, height)
    col_idx = np.linspace(0, small_w - 1, width)

    # Row-wise interpolation
    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, small_h - 1)
    rw = row_idx - r0

    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, small_w - 1)
    cw = col_idx - c0

    # Bilinear
    top = small[r0][:, c0] * (1 - cw) + small[r0][:, c1] * cw
    bot = small[r1][:, c0] * (1 - cw) + small[r1][:, c1] * cw
    result = top * (1 - rw[:, np.newaxis]) + bot * rw[:, np.newaxis]

    return result


# ── Wire rendering with specular highlight ────────────────────────────────────


def _render_wire_on_image(
    image: np.ndarray,
    profile: BondWireProfile,
    rng: Generator,
) -> np.ndarray:
    """Draw a single wire onto *image* with specular shading.

    The wire path follows a Bezier curve.  A simple cylindrical specular
    model is used: the wire is brightest at the centre and falls off toward
    the edges, with a specular highlight strip running along its length.

    Args:
        image:   Mutable RGB image (float64, 0-255).
        profile: Wire profile.
        rng:     Generator for noise.

    Returns:
        Modified *image* (same array, mutated in place for efficiency).
    """
    h, w = image.shape[:2]
    cps = _bezier_control_points(profile)
    curve = _evaluate_bezier(cps, n_samples=400)  # (N, 2)

    base_colour = _MATERIAL_COLOUR.get(profile.material, _MATERIAL_COLOUR["gold"])
    radius = profile.diameter / 2.0

    # Specular light direction (slightly off-centre for realism)
    spec_offset = rng.uniform(-0.3, 0.3) * radius

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = np.stack([xx, yy], axis=-1)  # (H, W, 2)

    # Vectorised min-distance computation
    diff = coords[:, :, np.newaxis, :] - curve[np.newaxis, np.newaxis, :, :]  # (H,W,N,2)
    dists_sq = (diff ** 2).sum(axis=-1)  # (H, W, N)
    min_idx = dists_sq.argmin(axis=-1)  # (H, W)
    min_dist = np.sqrt(dists_sq.min(axis=-1))  # (H, W)

    wire_mask = min_dist <= radius

    if not wire_mask.any():
        return image

    # Shading: Lambertian falloff from centre + specular strip
    # Normalised distance from wire centre: 0 at centre, 1 at edge
    norm_dist = np.where(wire_mask, min_dist / max(radius, 1e-6), 1.0)

    # Diffuse component
    diffuse = np.clip(1.0 - norm_dist ** 2, 0.0, 1.0)

    # Specular: narrow Gaussian along the centre
    specular = np.exp(-((min_dist - spec_offset) ** 2) / (0.3 * radius ** 2 + 1e-6))
    specular = np.where(wire_mask, specular, 0.0)

    # Combined shading
    shade = 0.6 * diffuse + 0.4 * specular
    shade = np.clip(shade, 0.0, 1.0)

    # Apply colour
    for c in range(3):
        wire_pixels = wire_mask
        image[:, :, c] = np.where(
            wire_pixels,
            base_colour[c] * shade + rng.normal(0, 2, size=(h, w)),
            image[:, :, c],
        )

    return image


# ── Full AOI image composition ────────────────────────────────────────────────


def render_aoi_image(
    wire_profiles: List[BondWireProfile],
    defects: Optional[List[Optional[str]]] = None,
    height: int = 512,
    width: int = 512,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Render a complete synthetic AOI image with wires and background.

    Args:
        wire_profiles: List of :class:`BondWireProfile` to draw.
        defects:       Parallel list of defect type strings (``"bend"``,
                       ``"break"``, ``"lift"``, or ``None`` for healthy).
                       If *None*, all wires are treated as healthy.
        height:        Image height.
        width:         Image width.
        rng:           NumPy Generator for reproducibility.

    Returns:
        ``uint8`` RGB image of shape ``(height, width, 3)``.
    """
    rng = _ensure_rng(rng)

    image = render_background(height, width, rng).astype(np.float64)

    if defects is None:
        defects = [None] * len(wire_profiles)

    for profile, defect_type in zip(wire_profiles, defects):
        image = _render_wire_on_image(image, profile, rng)

        # Add visual cues for defect regions (subtle discolouration / shadow)
        if defect_type == "break":
            # Dark shadow at break region (centre of wire)
            mask = render_wire_mask(profile, height, width)
            shadow = (mask > 0).astype(np.float64) * rng.uniform(10, 30)
            image[:, :, 0] = np.clip(image[:, :, 0] - shadow * 0.5, 0, 255)
            image[:, :, 1] = np.clip(image[:, :, 1] - shadow * 0.3, 0, 255)
        elif defect_type == "lift":
            # Slight blue-ish tint near lifted end
            mask = render_wire_mask(profile, height, width)
            tint = (mask > 0).astype(np.float64) * rng.uniform(5, 15)
            image[:, :, 2] = np.clip(image[:, :, 2] + tint, 0, 255)

    return np.clip(image, 0, 255).astype(np.uint8)
