"""Tests for UDM Epic 9 — Synthetic Crack Generation: geometry, types, rendering, transfer, dataset, metrics."""

from __future__ import annotations

import numpy as np
import torch

from udm_epic9.models.crack_geometry import (
    CrackProfile,
    generate_branching_crack,
    generate_crack_network,
    generate_crack_path,
    render_crack_mask,
)
from udm_epic9.models.crack_types import (
    delamination_crack,
    die_crack,
    mold_crack,
    substrate_crack,
)
from udm_epic9.rendering.usm_renderer import (
    generate_synthetic_usm_with_cracks,
    render_crack_on_usm,
)
from udm_epic9.domain_transfer.usm_to_rgb import USMtoRGBTransfer, mask_to_image
from udm_epic9.data.crack_dataset import CrackDataset
from udm_epic9.evaluation.crack_metrics import crack_detection_rate, crack_length_error


class TestCrackGeometry:
    """Crack path and mask generation."""

    def test_path_shape(self):
        path = generate_crack_path((0, 0), (100, 100), roughness=0.3, n_points=50)
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert len(path) >= 3  # at least start, mid, end

    def test_path_starts_near_start(self):
        path = generate_crack_path((10, 20), (200, 300), roughness=0.2, n_points=30)
        assert np.allclose(path[0], [10, 20], atol=0.1)

    def test_path_ends_near_end(self):
        path = generate_crack_path((10, 20), (200, 300), roughness=0.2, n_points=30)
        assert np.allclose(path[-1], [200, 300], atol=0.1)

    def test_mask_rendering_shape(self):
        rng = np.random.default_rng(42)
        path = generate_crack_path((10, 10), (200, 200), rng=rng)
        mask = render_crack_mask([path], height=256, width=256, rng=rng)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8

    def test_mask_has_nonzero_pixels(self):
        rng = np.random.default_rng(42)
        path = generate_crack_path((10, 10), (200, 200), rng=rng)
        mask = render_crack_mask([path], height=256, width=256, rng=rng)
        assert mask.sum() > 0

    def test_branching_creates_multiple_paths(self):
        rng = np.random.default_rng(42)
        paths = generate_branching_crack(
            origin=(128, 128), n_branches=3, max_depth=2,
            rng=rng, height=256, width=256,
        )
        assert len(paths) >= 3  # at least the root branches

    def test_network_creates_paths(self):
        rng = np.random.default_rng(42)
        paths = generate_crack_network(256, 256, n_cracks=5, rng=rng)
        assert len(paths) == 5

    def test_crack_profile_dataclass(self):
        p = CrackProfile(start_xy=(0, 0), end_xy=(10, 10), crack_type="linear")
        assert p.crack_type == "linear"
        assert p.width_range == (1, 4)


class TestCrackTypes:
    """Semiconductor-specific crack generators."""

    def test_die_crack_returns_valid(self):
        mask, meta = die_crack(256, 256, rng=np.random.default_rng(42))
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert meta["crack_type"] == "die_crack"
        assert mask.max() == 255 or mask.max() == 0

    def test_substrate_crack_returns_valid(self):
        mask, meta = substrate_crack(256, 256, rng=np.random.default_rng(42))
        assert mask.shape == (256, 256)
        assert meta["crack_type"] == "substrate_crack"

    def test_mold_crack_returns_valid(self):
        mask, meta = mold_crack(256, 256, rng=np.random.default_rng(42))
        assert mask.shape == (256, 256)
        assert meta["crack_type"] == "mold_crack"

    def test_delamination_crack_returns_valid(self):
        mask, meta = delamination_crack(256, 256, rng=np.random.default_rng(42))
        assert mask.shape == (256, 256)
        assert meta["crack_type"] == "delamination_crack"

    def test_masks_are_binary(self):
        """All crack masks should only contain 0 and 255."""
        for gen_fn in [die_crack, substrate_crack, mold_crack, delamination_crack]:
            mask, _ = gen_fn(128, 128, rng=np.random.default_rng(7))
            unique = set(np.unique(mask))
            assert unique.issubset({0, 255}), f"{gen_fn.__name__} has values {unique}"


class TestUSMRenderer:
    """USM crack rendering."""

    def test_render_crack_on_usm_shape_dtype(self):
        rng = np.random.default_rng(42)
        bg = np.full((128, 128), 0.4, dtype=np.float32)
        mask, _ = die_crack(128, 128, rng=rng)
        result = render_crack_on_usm(bg, mask, rng=rng)
        assert result.shape == (128, 128)
        assert result.dtype == np.float32

    def test_crack_region_is_brighter(self):
        rng = np.random.default_rng(42)
        bg = np.full((128, 128), 0.3, dtype=np.float32)
        mask, _ = substrate_crack(128, 128, rng=rng)
        result = render_crack_on_usm(bg, mask, crack_intensity=0.5, rng=rng)
        crack_pixels = result[mask > 0]
        bg_pixels = result[mask == 0]
        if len(crack_pixels) > 0 and len(bg_pixels) > 0:
            assert crack_pixels.mean() > bg_pixels.mean()

    def test_generate_synthetic_usm_returns_tuple(self):
        rng = np.random.default_rng(42)
        img, mask, meta = generate_synthetic_usm_with_cracks(
            height=128, width=128, n_cracks=2, rng=rng,
        )
        assert img.shape == (128, 128)
        assert mask.shape == (128, 128)
        assert img.dtype == np.float32
        assert mask.dtype == np.uint8
        assert meta["domain"] == "usm"
        assert meta["n_cracks"] == 2


class TestDomainTransfer:
    """USM to RGB transfer and mask-to-image generation."""

    def test_colormap_produces_3_channels(self):
        usm = np.random.rand(64, 64).astype(np.float32)
        transfer = USMtoRGBTransfer(method="colormap")
        rgb = transfer.transfer(usm)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_transfer_with_cracks(self):
        rng = np.random.default_rng(42)
        usm = np.full((64, 64), 0.4, dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:25, 10:50] = 255
        rgb = USMtoRGBTransfer().transfer_with_cracks(usm, mask)
        assert rgb.shape == (64, 64, 3)

    def test_mask_to_image_usm(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:35, 20:100] = 255
        result = mask_to_image(mask, target_domain="usm")
        assert result.shape == (128, 128)
        assert result.dtype == np.float32

    def test_mask_to_image_rgb(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:35, 20:100] = 255
        result = mask_to_image(mask, target_domain="rgb")
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_invalid_method_raises(self):
        import pytest
        with pytest.raises(ValueError):
            USMtoRGBTransfer(method="invalid")


class TestCrackDataset:
    """CrackDataset returns correct keys and shapes."""

    def test_on_the_fly_returns_correct_keys(self):
        ds = CrackDataset(n_samples=5, image_size=(64, 64), domain="usm", seed=42)
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "crack_type" in sample
        assert "domain" in sample

    def test_on_the_fly_shapes_usm(self):
        ds = CrackDataset(n_samples=3, image_size=(64, 64), domain="usm", seed=42)
        sample = ds[0]
        assert sample["image"].shape == (1, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)

    def test_on_the_fly_shapes_rgb(self):
        ds = CrackDataset(n_samples=3, image_size=(64, 64), domain="rgb", seed=42)
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)

    def test_len(self):
        ds = CrackDataset(n_samples=10, image_size=(32, 32))
        assert len(ds) == 10


class TestCrackMetrics:
    """Crack detection and length metrics."""

    def test_perfect_detection_rate(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:15, 10:50] = 255
        rate = crack_detection_rate([mask], [mask])
        assert rate == 1.0

    def test_zero_detection_rate(self):
        true_mask = np.zeros((64, 64), dtype=np.uint8)
        true_mask[10:15, 10:50] = 255
        pred_mask = np.zeros((64, 64), dtype=np.uint8)
        rate = crack_detection_rate([pred_mask], [true_mask])
        assert rate == 0.0

    def test_no_crack_images_ignored(self):
        """Images with no ground-truth cracks should not affect detection rate."""
        empty = np.zeros((64, 64), dtype=np.uint8)
        rate = crack_detection_rate([empty], [empty])
        assert rate == 1.0  # vacuously true

    def test_crack_length_error_zero_for_identical(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:22, 10:50] = 255
        err = crack_length_error(mask, mask)
        assert err == 0.0

    def test_crack_length_error_positive_for_mismatch(self):
        true_mask = np.zeros((64, 64), dtype=np.uint8)
        true_mask[20:22, 10:50] = 255
        pred_mask = np.zeros((64, 64), dtype=np.uint8)
        pred_mask[20:22, 10:30] = 255
        err = crack_length_error(pred_mask, true_mask)
        assert err > 0.0
