"""Tests for UDM Epic 2 — crop extraction, edges, paste, quality filter."""

import cv2
import numpy as np

from udm_epic2.conditioning.edges import edge_map_from_mask
from udm_epic2.dataset.crops import CropConfig, extract_crops_for_pair, process_crop_dataset
from udm_epic2.dataset.hf_export import export_hf_style_folder
from udm_epic2.integration.paste import paste_defect_on_background
from udm_epic2.quality.filter import laplacian_variance, passes_quality_gate


class TestEdgeMap:
    def test_empty_mask(self):
        m = np.zeros((16, 16), dtype=np.uint8)
        e = edge_map_from_mask(m)
        assert e.shape == m.shape
        assert e.sum() == 0

    def test_square_void_has_edges(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        m[16:48, 16:48] = 255
        e = edge_map_from_mask(m)
        assert e.max() > 0


class TestExtractCrops:
    def test_one_component(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        m = np.zeros((100, 100), dtype=np.uint8)
        m[40:60, 40:60] = 255
        cfg = CropConfig(min_component_area_px=10, padding_px=2)
        crops = list(extract_crops_for_pair(img, m, "test", cfg))
        assert len(crops) == 1
        ci, cm, meta = crops[0]
        assert ci.shape[0] > 0 and cm.shape[0] > 0
        assert meta["defect_class"] == "void"
        assert meta["component_area_px"] == 20 * 20

    def test_process_dataset_writes_manifest(self, tmp_path):
        img_dir = tmp_path / "im"
        msk_dir = tmp_path / "ms"
        img_dir.mkdir()
        msk_dir.mkdir()
        img = np.full((80, 80), 300, dtype=np.uint16)
        m = np.zeros((80, 80), dtype=np.uint8)
        m[20:50, 20:50] = 255
        cv2.imwrite(str(img_dir / "a.png"), img)
        cv2.imwrite(str(msk_dir / "a.png"), m)

        out = tmp_path / "out"
        cfg = CropConfig(min_component_area_px=50, padding_px=4)
        man = process_crop_dataset(
            img_dir, msk_dir, out, cfg, glob="*.png", write_edges=False
        )
        assert man.is_file()
        text = man.read_text()
        assert "crop_id" in text


class TestPaste:
    def test_alpha_blend(self):
        bg = np.zeros((100, 100), dtype=np.uint8)
        bg[:] = 100
        patch = np.full((40, 40), 200, dtype=np.uint8)
        m = np.zeros((40, 40), dtype=np.uint8)
        m[10:30, 10:30] = 255
        out = paste_defect_on_background(bg, patch, m, (50, 50), mode="alpha")
        assert out.shape == bg.shape


class TestQualityFilter:
    def test_laplacian(self):
        g = np.zeros((32, 32), dtype=np.uint8)
        g[8:24, 8:24] = 200
        v = laplacian_variance(g)
        assert v > 0
        ok, _ = passes_quality_gate(g, min_laplacian_var=1.0)
        assert ok


class TestHFExport:
    def test_export(self, tmp_path):
        import pandas as pd

        root = tmp_path / "crops"
        (root / "images" / "void").mkdir(parents=True)
        (root / "edges" / "void").mkdir(parents=True)
        img = np.full((32, 32), 128, dtype=np.uint8)
        edg = np.full((32, 32), 255, dtype=np.uint8)
        cv2.imwrite(str(root / "images" / "void" / "a.png"), img)
        cv2.imwrite(str(root / "edges" / "void" / "a.png"), edg)

        pd.DataFrame(
            [
                {
                    "image_relpath": "images/void/a.png",
                    "edge_relpath": "edges/void/a.png",
                }
            ]
        ).to_csv(root / "manifest.csv", index=False)

        out = tmp_path / "hf"
        meta = export_hf_style_folder(root, out)
        assert meta.is_file()
        assert (out / "train" / "image" / "000000.png").is_file()

