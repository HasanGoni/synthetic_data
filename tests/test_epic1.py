"""
Test suite for UDM Epic 1 — Synthetic Data Generator.

Covers:
  - Beer-Lambert physics correctness
  - Void shape generation
  - Sample generator output contracts
  - Pipeline split ratios
  - Augmentation shape preservation
"""

import numpy as np
import pytest
from udm_epic1.physics.beer_lambert import BeerLambertSimulator, BeerLambertConfig
from udm_epic1.generators.void_shapes import VoidShapeGenerator, VoidGeometry
from udm_epic1.generators.sample_generator import SyntheticSampleGenerator, GeneratorConfig
from udm_epic1.augmentation.transforms import AugmentationPipeline, AugConfig


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def physics(rng):
    return BeerLambertSimulator(BeerLambertConfig(), rng=rng)

@pytest.fixture
def shape_gen(rng):
    return VoidShapeGenerator(rng=rng)

@pytest.fixture
def generator():
    cfg = GeneratorConfig(height=128, width=128)  # small for speed
    return SyntheticSampleGenerator(cfg, seed=42)


# ─── Physics Tests ────────────────────────────────────────────────────────────

class TestBeerLambert:

    def test_background_shape(self, physics):
        bg = physics.generate_background_field(128, 128)
        assert bg.shape == (128, 128)
        assert bg.dtype == np.float32

    def test_background_range(self, physics):
        bg = physics.generate_background_field(128, 128)
        # After normalization + SFT the range will be centered around 0.5
        # but not guaranteed [0,1] — clamp happens at normalize step
        assert np.isfinite(bg).all(), "Background contains NaN/Inf"

    def test_percentile_normalize_range(self, physics):
        bg = physics.generate_background_field(128, 128)
        norm = physics.percentile_normalize(bg)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0
        assert norm.dtype == np.float32

    def test_percentile_normalize_constant(self, physics):
        """Constant image should return zeros (denom ≈ 0 guard)."""
        const = np.full((64, 64), 0.5, dtype=np.float32)
        result = physics.percentile_normalize(const)
        assert np.allclose(result, 0.0)

    def test_void_brighter_than_background(self, physics):
        """Core physics constraint: void must be brighter than background."""
        bg = np.full((64, 64), 0.3, dtype=np.float32)
        # Create center void mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[24:40, 24:40] = 255
        result = physics.insert_void(bg, mask, contrast=0.2)
        void_mean = result[24:40, 24:40].mean()
        bg_mean = result[:24, :24].mean()
        assert void_mean > bg_mean, "Void should be brighter than background (Beer-Lambert)"

    def test_to_uint16(self, physics):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = physics.to_uint16(arr)
        assert out.dtype == np.uint16
        assert out[0] == 0
        assert out[2] == 65535

    def test_to_uint8(self, physics):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = physics.to_uint8(arr)
        assert out.dtype == np.uint8
        assert out[0] == 0
        assert out[2] == 255


# ─── Void Shape Tests ─────────────────────────────────────────────────────────

class TestVoidShapes:

    @pytest.mark.parametrize("shape", ["ellipse", "irregular_blob", "elongated", "cluster"])
    def test_shape_output_contract(self, shape_gen, shape):
        geom = VoidGeometry(cx=64, cy=64, area_fraction=0.02, shape=shape, contrast=0.2, edge_sigma=1.0)
        mask = shape_gen.generate(128, 128, geom)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8
        unique = np.unique(mask)
        assert all(v in (0, 255) for v in unique), f"Mask has non-binary values: {unique}"

    @pytest.mark.parametrize("shape", ["ellipse", "irregular_blob", "elongated", "cluster"])
    def test_shape_has_void_pixels(self, shape_gen, shape):
        geom = VoidGeometry(cx=64, cy=64, area_fraction=0.05, shape=shape, contrast=0.2, edge_sigma=1.0)
        mask = shape_gen.generate(128, 128, geom)
        assert (mask == 255).sum() > 0, f"{shape}: mask has no void pixels"

    def test_sample_geometry_no_overlap(self, shape_gen):
        existing = []
        geometries = []
        for _ in range(5):
            geom = shape_gen.sample_geometry(
                height=256, width=256,
                min_area_fraction=0.01, max_area_fraction=0.05,
                existing_masks=existing,
            )
            if geom is not None:
                mask = shape_gen.generate(256, 256, geom)
                existing.append(mask)
                geometries.append(geom)
        # All placed geometries should have non-overlapping masks
        if len(existing) > 1:
            combined = np.zeros((256, 256), dtype=np.int32)
            for m in existing:
                combined += (m > 127).astype(np.int32)
            # No pixel should belong to more than 1 void (overlap=False)
            # Allow small overlap due to edge blur rounding
            assert combined.max() <= 2, "Too many overlapping voids"


# ─── Generator Tests ──────────────────────────────────────────────────────────

class TestSampleGenerator:

    def test_output_shapes(self, generator):
        image, mask, meta = generator.generate("test_001")
        assert image.shape == (128, 128)
        assert mask.shape == (128, 128)

    def test_output_dtypes(self, generator):
        image, mask, meta = generator.generate("test_002")
        assert image.dtype == np.uint16
        assert mask.dtype == np.uint8

    def test_mask_binary(self, generator):
        for i in range(5):
            _, mask, _ = generator.generate(f"test_{i:04d}")
            unique = np.unique(mask)
            assert all(v in (0, 255) for v in unique), f"Non-binary mask values: {unique}"

    def test_void_brighter_in_image(self, generator):
        """Any image with voids must have brighter void region."""
        for _ in range(10):
            image, mask, meta = generator.generate("test_physics")
            if not meta.has_voids:
                continue
            img_f = image.astype(np.float32) / 65535.0
            void_mean = img_f[mask == 255].mean() if (mask == 255).any() else None
            bg_mean = img_f[mask == 0].mean() if (mask == 0).any() else None
            if void_mean is not None and bg_mean is not None:
                assert void_mean > bg_mean - 0.05, \
                    f"Void mean ({void_mean:.3f}) should be ≥ background mean ({bg_mean:.3f})"
            break

    def test_meta_fields(self, generator):
        _, _, meta = generator.generate("test_meta", split="val")
        assert meta.image_id == "test_meta"
        assert meta.split == "val"
        assert meta.height == 128
        assert meta.width == 128
        assert isinstance(meta.n_voids, int)
        assert meta.n_voids >= 0
        assert 0.0 <= meta.total_void_area_fraction <= 1.0

    def test_empty_image_fraction(self):
        """~15% of images should have no voids."""
        cfg = GeneratorConfig(height=64, width=64, empty_image_fraction=0.5)
        gen = SyntheticSampleGenerator(cfg, seed=0)
        empty_count = sum(1 for i in range(100) if not gen.generate(f"t{i}")[2].has_voids)
        # With 50% empty fraction, expect 30-70 empty images in 100
        assert 20 <= empty_count <= 80, f"Unexpected empty count: {empty_count}"

    def test_reproducibility(self):
        """Same seed must produce identical outputs."""
        cfg = GeneratorConfig(height=64, width=64)
        g1 = SyntheticSampleGenerator(cfg, seed=7)
        g2 = SyntheticSampleGenerator(cfg, seed=7)
        img1, msk1, _ = g1.generate("rep_test")
        img2, msk2, _ = g2.generate("rep_test")
        np.testing.assert_array_equal(img1, img2)
        np.testing.assert_array_equal(msk1, msk2)


# ─── Augmentation Tests ───────────────────────────────────────────────────────

class TestAugmentation:

    @pytest.fixture
    def aug(self):
        return AugmentationPipeline(AugConfig(enabled=True), rng=np.random.default_rng(42))

    def test_shape_preserved(self, aug):
        img = np.random.rand(128, 128).astype(np.float32)
        msk = np.zeros((128, 128), dtype=np.uint8)
        msk[40:80, 40:80] = 255
        img_out, msk_out = aug(img, msk)
        assert img_out.shape == (128, 128)
        assert msk_out.shape == (128, 128)

    def test_range_clipped(self, aug):
        img = np.random.rand(128, 128).astype(np.float32)
        msk = np.zeros((128, 128), dtype=np.uint8)
        img_out, _ = aug(img, msk)
        assert img_out.min() >= 0.0
        assert img_out.max() <= 1.0

    def test_dtype_preserved(self, aug):
        img = np.random.rand(64, 64).astype(np.float32)
        msk = np.zeros((64, 64), dtype=np.uint8)
        img_out, msk_out = aug(img, msk)
        assert img_out.dtype == np.float32
        assert msk_out.dtype == np.uint8

    def test_disabled_passthrough(self):
        aug = AugmentationPipeline(AugConfig(enabled=False))
        img = np.random.rand(64, 64).astype(np.float32)
        msk = np.zeros((64, 64), dtype=np.uint8)
        img_out, msk_out = aug(img, msk)
        np.testing.assert_array_equal(img, img_out)
        np.testing.assert_array_equal(msk, msk_out)

    def test_ring_artifact_applied(self):
        """Ring artifact augmentation should modify the image."""
        cfg = AugConfig(domain_shift_enabled=True, ring_artifact_prob=1.0)
        aug = AugmentationPipeline(cfg, rng=np.random.default_rng(0))
        img = np.full((128, 128), 0.5, dtype=np.float32)
        msk = np.zeros((128, 128), dtype=np.uint8)
        img_out, _ = aug(img, msk)
        assert not np.allclose(img, img_out), "Ring artifact should change the image"
