"""Tests for UDM Epic 8 -- Universal Model Support: registry, export, merge, cross-modality report."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from udm_epic8.registry.modality_registry import ModalityRegistry, registry
from udm_epic8.export.dataset_export import (
    export_to_coco,
    export_to_hf,
    export_to_yolo,
    merge_datasets,
)
from udm_epic8.evaluation.cross_modality import (
    compare_real_vs_synthetic,
    cross_modality_report,
)
from udm_epic8.pipeline.unified import UnifiedPipelineConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="test_epic8_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_dataset(tmp_dir):
    """Create a minimal synthetic dataset with images and masks."""
    images_dir = tmp_dir / "images"
    masks_dir = tmp_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255  # defect region
        Image.fromarray(img).save(str(images_dir / f"img_{i:03d}.png"))
        Image.fromarray(mask).save(str(masks_dir / f"img_{i:03d}_mask.png"))

    return tmp_dir


# ── Registry tests ────────────────────────────────────────────────────────────


class TestModalityRegistry:
    """ModalityRegistry register / list / get / generate."""

    def test_register_and_list(self):
        reg = ModalityRegistry()
        reg.register("test_mod", lambda **kw: Path(kw["output_dir"]), dict)
        assert "test_mod" in reg.list_modalities()

    def test_get_registered(self):
        reg = ModalityRegistry()
        fn = lambda **kw: Path(kw["output_dir"])
        reg.register("my_mod", fn, dict)
        got_fn, got_cls = reg.get("my_mod")
        assert got_fn is fn
        assert got_cls is dict

    def test_get_missing_raises(self):
        reg = ModalityRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_list_sorted(self):
        reg = ModalityRegistry()
        reg.register("zebra", lambda **kw: Path("."), dict)
        reg.register("alpha", lambda **kw: Path("."), dict)
        assert reg.list_modalities() == ["alpha", "zebra"]

    def test_generate_calls_fn(self, tmp_dir):
        reg = ModalityRegistry()
        called_with = {}

        def mock_gen(config, n_samples, output_dir):
            called_with.update(config=config, n_samples=n_samples, output_dir=output_dir)
            return Path(output_dir) / "manifest.json"

        reg.register("mock", mock_gen, dict)
        result = reg.generate("mock", config={"key": "val"}, n_samples=10, output_dir=str(tmp_dir))
        assert called_with["n_samples"] == 10
        assert called_with["config"] == {"key": "val"}
        assert result == tmp_dir / "manifest.json"

    def test_global_registry_has_all_epics(self):
        expected = {"xray", "controlnet", "cyclegan", "dann", "active", "aoi_wire", "chromasense"}
        registered = set(registry.list_modalities())
        assert expected.issubset(registered), f"Missing: {expected - registered}"


# ── Export tests ──────────────────────────────────────────────────────────────


class TestExportCOCO:
    """COCO format export correctness."""

    def test_produces_valid_json(self, sample_dataset, tmp_dir):
        out = tmp_dir / "coco_out" / "annotations.json"
        result = export_to_coco(str(sample_dataset), str(out), modality="xray")
        assert result.exists()
        data = json.loads(result.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 5

    def test_annotations_have_bbox(self, sample_dataset, tmp_dir):
        out = tmp_dir / "coco_out" / "annotations.json"
        export_to_coco(str(sample_dataset), str(out))
        data = json.loads(out.read_text())
        for ann in data["annotations"]:
            assert "bbox" in ann
            assert "area" in ann
            assert ann["area"] > 0

    def test_missing_images_dir_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            export_to_coco(str(tmp_dir / "nonexistent"), str(tmp_dir / "out.json"))


class TestExportYOLO:
    """YOLO format export correctness."""

    def test_produces_labels(self, sample_dataset, tmp_dir):
        out = tmp_dir / "yolo_out"
        result = export_to_yolo(str(sample_dataset), str(out))
        labels_dir = result / "labels"
        assert labels_dir.is_dir()
        label_files = list(labels_dir.glob("*.txt"))
        assert len(label_files) == 5

    def test_label_format(self, sample_dataset, tmp_dir):
        out = tmp_dir / "yolo_out"
        export_to_yolo(str(sample_dataset), str(out))
        label_files = sorted((out / "labels").glob("*.txt"))
        content = label_files[0].read_text().strip()
        parts = content.split()
        assert len(parts) == 5  # class x_center y_center width height
        assert parts[0] == "0"  # class 0
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0

    def test_data_yaml(self, sample_dataset, tmp_dir):
        out = tmp_dir / "yolo_out"
        export_to_yolo(str(sample_dataset), str(out))
        assert (out / "data.yaml").exists()


class TestExportHF:
    """HuggingFace format export correctness."""

    def test_produces_metadata(self, sample_dataset, tmp_dir):
        out = tmp_dir / "hf_out"
        result = export_to_hf(str(sample_dataset), str(out))
        assert (result / "metadata.jsonl").exists()
        lines = (result / "metadata.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            entry = json.loads(line)
            assert "file_name" in entry


# ── Merge tests ───────────────────────────────────────────────────────────────


class TestMergeDatasets:
    """Merge multiple modality directories."""

    def test_merge_two_datasets(self, tmp_dir):
        # Create two small datasets
        for prefix in ("mod_a", "mod_b"):
            d = tmp_dir / prefix
            (d / "images").mkdir(parents=True)
            (d / "masks").mkdir(parents=True)
            for i in range(3):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(img).save(str(d / "images" / f"{prefix}_{i}.png"))
                mask = np.zeros((32, 32), dtype=np.uint8)
                Image.fromarray(mask).save(str(d / "masks" / f"{prefix}_{i}_mask.png"))

        merged_dir = tmp_dir / "merged"
        manifest = merge_datasets(
            [str(tmp_dir / "mod_a"), str(tmp_dir / "mod_b")],
            str(merged_dir),
        )
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["total_images"] == 6
        assert len(list((merged_dir / "images").glob("*.png"))) == 6


# ── Cross-modality report tests ──────────────────────────────────────────────


class TestCrossModalityReport:
    """cross_modality_report DataFrame construction."""

    def test_basic_report(self):
        results = {
            "xray": {"f1": 0.91, "iou": 0.84, "n_samples": 500},
            "aoi": {"f1": 0.87, "iou": 0.79, "n_samples": 300},
        }
        df = cross_modality_report(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "modality" in df.columns
        assert "f1" in df.columns

    def test_empty_results(self):
        df = cross_modality_report({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_missing_metric_is_nan(self):
        results = {
            "xray": {"f1": 0.9, "fid": 12.3},
            "aoi": {"f1": 0.8},
        }
        df = cross_modality_report(results)
        aoi_row = df[df["modality"] == "aoi"].iloc[0]
        assert pd.isna(aoi_row["fid"])

    def test_sorted_by_modality(self):
        results = {
            "zebra": {"f1": 0.5},
            "alpha": {"f1": 0.9},
        }
        df = cross_modality_report(results)
        assert list(df["modality"]) == ["alpha", "zebra"]


# ── Pipeline config tests ────────────────────────────────────────────────────


class TestUnifiedPipelineConfig:
    """UnifiedPipelineConfig validation."""

    def test_default_config(self):
        cfg = UnifiedPipelineConfig()
        assert cfg.modalities == ["xray"]
        assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            UnifiedPipelineConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_samples_for_default(self):
        cfg = UnifiedPipelineConfig(total_samples=200)
        assert cfg.samples_for("xray") == 200

    def test_samples_for_override(self):
        cfg = UnifiedPipelineConfig(
            per_modality_config={"xray": {"samples": 50}},
            total_samples=200,
        )
        assert cfg.samples_for("xray") == 50
        assert cfg.samples_for("aoi") == 200  # falls back
