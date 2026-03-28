"""Tests for UDM Epic 7 — Chromasense: spectral config, reflectance, defects, rendering, dataset."""

import math

import numpy as np
import torch

from udm_epic7.spectral.wavelength_model import (
    SpectralConfig,
    default_spectral_config,
    material_reflectance,
)
from udm_epic7.spectral.defect_spectra import (
    delamination_spectrum,
    contamination_spectrum,
    oxidation_spectrum,
)
from udm_epic7.rendering.spectral_renderer import (
    render_spectral_image,
    render_single_wavelength,
    spectral_to_rgb,
)
from udm_epic7.data.dataset import SpectralDataset
from udm_epic7.evaluation.spectral_metrics import (
    spectral_angle_mapper,
    spectral_anomaly_score,
)


# ── Spectral Config ──────────────────────────────────────────────────────


class TestSpectralConfig:
    """SpectralConfig dataclass and default factory."""

    def test_default_config_wavelengths(self):
        cfg = default_spectral_config()
        assert cfg.wavelengths == [450.0, 550.0, 650.0, 850.0]

    def test_default_config_n_channels(self):
        cfg = default_spectral_config()
        assert cfg.n_channels == 4

    def test_default_config_materials(self):
        cfg = default_spectral_config()
        expected = {"copper", "mold_compound", "oxidized_copper", "silicon", "solder"}
        assert set(cfg.material_names) == expected

    def test_custom_config(self):
        cfg = SpectralConfig(
            wavelengths=[400.0, 600.0],
            material_spectra={"test_mat": {400.0: 0.5, 600.0: 0.7}},
        )
        assert cfg.n_channels == 2
        assert cfg.material_names == ["test_mat"]


# ── Material Reflectance ─────────────────────────────────────────────────


class TestMaterialReflectance:
    """Reflectance look-up and interpolation."""

    def test_exact_wavelength_lookup(self):
        ref = material_reflectance("copper", 450.0)
        assert ref == 0.18

    def test_interpolation(self):
        # 500 nm is between 450 and 550, copper goes 0.18 -> 0.45
        ref = material_reflectance("copper", 500.0)
        expected = 0.18 + 0.5 * (0.45 - 0.18)  # linear at midpoint
        assert abs(ref - expected) < 1e-6

    def test_clamped_below_range(self):
        ref = material_reflectance("copper", 300.0)
        assert ref == 0.18  # clamped to lowest wavelength value

    def test_clamped_above_range(self):
        ref = material_reflectance("copper", 1000.0)
        assert ref == 0.83  # clamped to highest wavelength value

    def test_unknown_material_raises(self):
        try:
            material_reflectance("unobtanium", 450.0)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_all_materials_have_valid_reflectance(self):
        cfg = default_spectral_config()
        for mat in cfg.material_names:
            for wl in cfg.wavelengths:
                ref = material_reflectance(mat, wl, cfg)
                assert 0.0 <= ref <= 1.0, f"{mat} @ {wl} nm: {ref}"


# ── Defect Spectra ───────────────────────────────────────────────────────


class TestDefectSpectra:
    """Defect spectral modification functions."""

    def test_delamination_zero_severity_matches_base(self):
        cfg = default_spectral_config()
        spec = delamination_spectrum("copper", severity=0.0, config=cfg)
        for wl in cfg.wavelengths:
            base = material_reflectance("copper", wl, cfg)
            assert abs(spec[wl] - base) < 0.01

    def test_delamination_high_severity_reduces_reflectance(self):
        cfg = default_spectral_config()
        spec = delamination_spectrum("copper", severity=0.9, config=cfg)
        for wl in cfg.wavelengths:
            base = material_reflectance("copper", wl, cfg)
            # Delamination should generally reduce reflectance
            assert spec[wl] <= base + 0.1  # allow small fringe offset

    def test_contamination_returns_all_wavelengths(self):
        cfg = default_spectral_config()
        spec = contamination_spectrum("flux_residue", 0.5, "copper", cfg)
        assert set(spec.keys()) == set(cfg.wavelengths)

    def test_contamination_blends_between_materials(self):
        cfg = default_spectral_config()
        spec_low = contamination_spectrum("dust", 0.1, "copper", cfg)
        spec_high = contamination_spectrum("dust", 0.9, "copper", cfg)
        # At high concentration, spectrum should differ more from base copper
        for wl in cfg.wavelengths:
            base = material_reflectance("copper", wl, cfg)
            diff_low = abs(spec_low[wl] - base)
            diff_high = abs(spec_high[wl] - base)
            assert diff_high >= diff_low - 0.01  # high conc deviates more

    def test_contamination_unknown_raises(self):
        try:
            contamination_spectrum("alien_goo", 0.5, "copper")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_oxidation_reduces_copper_reflectance(self):
        cfg = default_spectral_config()
        spec = oxidation_spectrum("copper", 0.8, cfg)
        for wl in cfg.wavelengths:
            base = material_reflectance("copper", wl, cfg)
            assert spec[wl] < base

    def test_oxidation_zero_preserves_base(self):
        cfg = default_spectral_config()
        spec = oxidation_spectrum("silicon", 0.0, cfg)
        for wl in cfg.wavelengths:
            base = material_reflectance("silicon", wl, cfg)
            assert abs(spec[wl] - base) < 1e-6

    def test_defect_spectra_values_in_range(self):
        cfg = default_spectral_config()
        for sev in [0.0, 0.3, 0.7, 1.0]:
            specs = [
                delamination_spectrum("copper", severity=sev, config=cfg),
                contamination_spectrum("flux_residue", concentration=sev, base_material="copper", config=cfg),
                oxidation_spectrum("copper", oxidation_level=sev, config=cfg),
            ]
            for spec in specs:
                for wl, ref in spec.items():
                    assert 0.0 <= ref <= 1.0, f"{func.__name__} sev={severity} wl={wl}: {ref}"


# ── Rendering ────────────────────────────────────────────────────────────


class TestRendering:
    """Multi-spectral image rendering."""

    def test_render_spectral_image_shape(self):
        layout = np.zeros((64, 64), dtype=np.int32)
        layout[16:48, 16:48] = 1  # silicon region
        img = render_spectral_image(layout, height=64, width=64)
        assert img.shape == (4, 64, 64)
        assert img.dtype == np.float32

    def test_render_spectral_image_values_in_range(self):
        layout = np.ones((64, 64), dtype=np.int32)  # all silicon
        img = render_spectral_image(layout, height=64, width=64)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_render_with_defects(self):
        layout = np.full((64, 64), 2, dtype=np.int32)  # all copper
        defects = [{
            "type": "oxidation",
            "bbox": (10, 10, 30, 30),
            "severity": 0.8,
            "material": "copper",
        }]
        img = render_spectral_image(layout, defects=defects, height=64, width=64)
        assert img.shape == (4, 64, 64)

    def test_render_single_wavelength_shape(self):
        layout = np.zeros((64, 64), dtype=np.int32)
        ch = render_single_wavelength(layout, 550.0, height=64, width=64)
        assert ch.shape == (64, 64)

    def test_render_custom_config(self):
        cfg = SpectralConfig(
            wavelengths=[500.0, 700.0],
            material_spectra={"copper": {500.0: 0.4, 700.0: 0.8}},
        )
        layout = np.full((32, 32), 2, dtype=np.int32)  # copper label
        img = render_spectral_image(layout, config=cfg, height=32, width=32)
        assert img.shape == (2, 32, 32)

    def test_spectral_to_rgb_shape(self):
        img = np.random.rand(4, 64, 64).astype(np.float32)
        rgb = spectral_to_rgb(img)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_spectral_to_rgb_range(self):
        img = np.random.rand(4, 32, 32).astype(np.float32)
        rgb = spectral_to_rgb(img)
        assert rgb.min() >= 0
        assert rgb.max() <= 255


# ── Dataset ──────────────────────────────────────────────────────────────


class TestSpectralDataset:
    """SpectralDataset generation and access."""

    def test_dataset_length(self):
        ds = SpectralDataset(n_samples=10, height=32, width=32)
        assert len(ds) == 10

    def test_dataset_sample_keys(self):
        ds = SpectralDataset(n_samples=5, height=32, width=32)
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "defect_type" in sample

    def test_dataset_image_shape(self):
        ds = SpectralDataset(n_samples=5, height=64, width=64)
        sample = ds[0]
        assert sample["image"].shape == (4, 64, 64)
        assert sample["image"].dtype == torch.float32

    def test_dataset_mask_shape(self):
        ds = SpectralDataset(n_samples=5, height=64, width=64)
        sample = ds[0]
        assert sample["mask"].shape == (1, 64, 64)

    def test_dataset_defect_type_is_string(self):
        ds = SpectralDataset(n_samples=5, height=32, width=32)
        sample = ds[0]
        assert isinstance(sample["defect_type"], str)
        assert sample["defect_type"] in {"delamination", "contamination", "oxidation", "none"}

    def test_dataset_reproducibility(self):
        ds1 = SpectralDataset(n_samples=3, height=32, width=32, seed=123)
        ds2 = SpectralDataset(n_samples=3, height=32, width=32, seed=123)
        for i in range(3):
            assert torch.allclose(ds1[i]["image"], ds2[i]["image"])

    def test_dataset_custom_config(self):
        cfg = SpectralConfig(
            wavelengths=[500.0, 700.0],
            material_spectra={
                "mold_compound": {500.0: 0.1, 700.0: 0.1},
                "silicon": {500.0: 0.4, 700.0: 0.3},
                "copper": {500.0: 0.5, 700.0: 0.8},
                "solder": {500.0: 0.5, 700.0: 0.6},
            },
        )
        ds = SpectralDataset(n_samples=3, config=cfg, height=32, width=32)
        sample = ds[0]
        assert sample["image"].shape[0] == 2  # 2 wavelengths

    def test_dataset_repr(self):
        ds = SpectralDataset(n_samples=10, height=32, width=32)
        r = repr(ds)
        assert "SpectralDataset" in r
        assert "n=10" in r


# ── Spectral Metrics ─────────────────────────────────────────────────────


class TestSpectralMetrics:
    """SAM and anomaly scoring."""

    def test_sam_identical_spectra(self):
        a = np.array([0.3, 0.5, 0.7, 0.8])
        assert spectral_angle_mapper(a, a) < 1e-6

    def test_sam_orthogonal_spectra(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(spectral_angle_mapper(a, b) - math.pi / 2) < 1e-6

    def test_sam_scaled_spectra_equal(self):
        a = np.array([0.2, 0.4, 0.6])
        b = 3.0 * a
        assert spectral_angle_mapper(a, b) < 1e-6

    def test_sam_zero_vector(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.1, 0.2, 0.3])
        assert spectral_angle_mapper(a, b) == 0.0

    def test_anomaly_score_shape(self):
        img = np.random.rand(4, 32, 32).astype(np.float32)
        ref = np.array([0.3, 0.5, 0.7, 0.8])
        scores = spectral_anomaly_score(img, ref)
        assert scores.shape == (32, 32)
        assert scores.dtype == np.float32

    def test_anomaly_score_uniform_image(self):
        # Uniform image matching the reference should have near-zero anomaly
        ref = np.array([0.3, 0.5, 0.7, 0.8])
        img = np.tile(ref[:, None, None], (1, 16, 16)).astype(np.float32)
        scores = spectral_anomaly_score(img, ref)
        assert scores.max() < 1e-5

    def test_anomaly_score_mismatched_channels_raises(self):
        img = np.random.rand(4, 16, 16).astype(np.float32)
        ref = np.array([0.3, 0.5])  # wrong number of channels
        try:
            spectral_anomaly_score(img, ref)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
