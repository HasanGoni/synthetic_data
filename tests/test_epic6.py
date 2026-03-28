"""Tests for UDM Epic 6 — AOI Bond Wire Synthesis:
wire profiles, defect generators, mask rendering, dataset shapes, metrics."""

import numpy as np
import torch

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
from udm_epic6.rendering.aoi_renderer import render_aoi_image, render_background
from udm_epic6.data.dataset import BondWireDataset
from udm_epic6.evaluation.metrics import (
    compute_f1,
    compute_iou,
    wire_detection_rate,
    defect_classification_accuracy,
)


class TestWireProfile:
    """Bond wire profile generation."""

    def test_generate_returns_list(self):
        rng = np.random.default_rng(0)
        profiles = generate_wire_profile(rng, image_size=(256, 256), n_wires=3)
        assert isinstance(profiles, list)
        assert len(profiles) == 3

    def test_generate_random_count(self):
        rng = np.random.default_rng(1)
        profiles = generate_wire_profile(rng, image_size=(256, 256))
        assert 1 <= len(profiles) <= 5

    def test_profile_fields(self):
        rng = np.random.default_rng(2)
        profiles = generate_wire_profile(rng, image_size=(512, 512), n_wires=1)
        p = profiles[0]
        assert isinstance(p, BondWireProfile)
        assert len(p.start_xy) == 2
        assert len(p.end_xy) == 2
        assert p.loop_height > 0
        assert p.diameter > 0
        assert p.material in ("gold", "copper", "aluminum")

    def test_wire_endpoints_within_image(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            profiles = generate_wire_profile(rng, image_size=(256, 256), n_wires=3)
            for p in profiles:
                assert 0 <= p.start_xy[0] <= 256
                assert 0 <= p.start_xy[1] <= 256
                assert 0 <= p.end_xy[0] <= 256
                assert 0 <= p.end_xy[1] <= 256


class TestMaskRendering:
    """Wire mask rendering via Bezier curves."""

    def test_mask_shape(self):
        p = BondWireProfile(start_xy=(20, 64), end_xy=(236, 64), loop_height=30, diameter=3)
        mask = render_wire_mask(p, height=128, width=256)
        assert mask.shape == (128, 256)
        assert mask.dtype == np.uint8

    def test_mask_has_content(self):
        p = BondWireProfile(start_xy=(20, 64), end_xy=(236, 64), loop_height=30, diameter=4)
        mask = render_wire_mask(p, height=128, width=256)
        assert mask.max() == 255
        assert mask.sum() > 0

    def test_mask_binary(self):
        p = BondWireProfile(start_xy=(10, 50), end_xy=(200, 50), loop_height=20, diameter=3)
        mask = render_wire_mask(p, height=128, width=256)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_thicker_wire_more_pixels(self):
        p_thin = BondWireProfile(start_xy=(20, 64), end_xy=(236, 64), loop_height=30, diameter=2)
        p_thick = BondWireProfile(start_xy=(20, 64), end_xy=(236, 64), loop_height=30, diameter=6)
        mask_thin = render_wire_mask(p_thin, 128, 256)
        mask_thick = render_wire_mask(p_thick, 128, 256)
        assert mask_thick.sum() > mask_thin.sum()


class TestDefectGenerators:
    """Defect application to wire profiles."""

    def _base_profile(self):
        return BondWireProfile(
            start_xy=(50.0, 128.0),
            end_xy=(462.0, 128.0),
            loop_height=40.0,
            diameter=3.0,
            curvature=0.0,
            material="gold",
        )

    def test_bend_returns_profile(self):
        rng = np.random.default_rng(10)
        result = apply_bend_defect(self._base_profile(), severity=0.5, rng=rng)
        assert isinstance(result, BondWireProfile)

    def test_bend_modifies_geometry(self):
        rng = np.random.default_rng(11)
        original = self._base_profile()
        bent = apply_bend_defect(original, severity=0.7, rng=rng)
        # At least one geometric property should differ
        changed = (
            bent.loop_height != original.loop_height
            or bent.curvature != original.curvature
            or bent.start_xy != original.start_xy
        )
        assert changed

    def test_break_returns_two_fragments(self):
        rng = np.random.default_rng(12)
        frag1, frag2 = apply_break_defect(self._base_profile(), break_position=0.5, rng=rng)
        assert isinstance(frag1, BondWireProfile)
        assert isinstance(frag2, BondWireProfile)

    def test_break_fragments_shorter(self):
        rng = np.random.default_rng(13)
        original = self._base_profile()
        frag1, frag2 = apply_break_defect(original, break_position=0.5, rng=rng)
        orig_span = np.hypot(
            original.end_xy[0] - original.start_xy[0],
            original.end_xy[1] - original.start_xy[1],
        )
        f1_span = np.hypot(
            frag1.end_xy[0] - frag1.start_xy[0],
            frag1.end_xy[1] - frag1.start_xy[1],
        )
        f2_span = np.hypot(
            frag2.end_xy[0] - frag2.start_xy[0],
            frag2.end_xy[1] - frag2.start_xy[1],
        )
        assert f1_span < orig_span
        assert f2_span < orig_span

    def test_lift_returns_profile(self):
        rng = np.random.default_rng(14)
        result = apply_lift_defect(self._base_profile(), rng=rng)
        assert isinstance(result, BondWireProfile)

    def test_lift_increases_loop_height(self):
        rng = np.random.default_rng(15)
        original = self._base_profile()
        lifted = apply_lift_defect(original, lift_amount=20.0, rng=rng)
        assert lifted.loop_height > original.loop_height


class TestAOIRendering:
    """Full AOI image rendering."""

    def test_background_shape(self):
        bg = render_background(256, 256, rng=np.random.default_rng(0))
        assert bg.shape == (256, 256, 3)
        assert bg.dtype == np.uint8

    def test_render_image_shape(self):
        rng = np.random.default_rng(0)
        profiles = generate_wire_profile(rng, image_size=(128, 128), n_wires=2)
        img = render_aoi_image(profiles, height=128, width=128, rng=rng)
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8

    def test_render_with_defects(self):
        rng = np.random.default_rng(0)
        profiles = generate_wire_profile(rng, image_size=(128, 128), n_wires=2)
        img = render_aoi_image(profiles, defects=["bend", None], height=128, width=128, rng=rng)
        assert img.shape == (128, 128, 3)


class TestBondWireDataset:
    """On-the-fly dataset output shapes and types."""

    def test_len(self):
        ds = BondWireDataset(n_samples=10, image_size=(64, 64), seed=0)
        assert len(ds) == 10

    def test_getitem_keys(self):
        ds = BondWireDataset(n_samples=5, image_size=(64, 64), seed=0)
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "defect_type" in sample
        assert "metadata" in sample

    def test_image_shape(self):
        ds = BondWireDataset(n_samples=5, image_size=(64, 64), seed=0)
        sample = ds[0]
        assert sample["image"].shape == (3, 64, 64)
        assert sample["image"].dtype == torch.float32

    def test_mask_shape(self):
        ds = BondWireDataset(n_samples=5, image_size=(64, 64), seed=0)
        sample = ds[0]
        assert sample["mask"].shape == (1, 64, 64)
        assert sample["mask"].dtype == torch.float32

    def test_image_range(self):
        ds = BondWireDataset(n_samples=5, image_size=(64, 64), seed=0)
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_deterministic(self):
        ds1 = BondWireDataset(n_samples=3, image_size=(64, 64), seed=123)
        ds2 = BondWireDataset(n_samples=3, image_size=(64, 64), seed=123)
        assert torch.allclose(ds1[0]["image"], ds2[0]["image"])
        assert torch.allclose(ds1[0]["mask"], ds2[0]["mask"])

    def test_defect_type_valid(self):
        ds = BondWireDataset(n_samples=20, image_size=(64, 64), seed=0)
        valid_types = {"none", "bend", "break", "lift"}
        for i in range(20):
            assert ds[i]["defect_type"] in valid_types


class TestMetrics:
    """Wire-specific and pixel-level metrics."""

    def test_perfect_f1(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        assert compute_f1(pred, target) == 1.0

    def test_zero_f1(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.zeros(1, 1, 8, 8)
        assert compute_f1(pred, target) == 0.0

    def test_perfect_iou(self):
        pred = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 1, 8, 8)
        assert compute_iou(pred, target) == 1.0

    def test_wire_detection_rate_perfect(self):
        pred = [np.ones((8, 8), dtype=np.uint8)]
        true = [np.ones((8, 8), dtype=np.uint8)]
        wire = [np.ones((8, 8), dtype=np.uint8)]
        assert wire_detection_rate(pred, true, wire) == 1.0

    def test_wire_detection_rate_empty(self):
        assert wire_detection_rate([], [], []) == 1.0

    def test_defect_classification_perfect(self):
        pred = ["bend", "break", "none"]
        true = ["bend", "break", "none"]
        assert defect_classification_accuracy(pred, true) == 1.0

    def test_defect_classification_partial(self):
        pred = ["bend", "break", "none"]
        true = ["bend", "lift", "none"]
        acc = defect_classification_accuracy(pred, true)
        assert abs(acc - 2.0 / 3.0) < 1e-6

    def test_defect_classification_empty(self):
        assert defect_classification_accuracy([], []) == 1.0
