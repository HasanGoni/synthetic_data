"""Tests for UDM Epic 5 — Active Domain Adaptation:
MC Dropout, coreset/combined selection, labeling sessions, learning curves."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from udm_epic5.uncertainty.mc_dropout import enable_mc_dropout, compute_entropy
from udm_epic5.selection.diversity import coreset_selection
from udm_epic5.selection.combined import combined_selection
from udm_epic5.labeling.session import LabelingSession, load_labeled_samples
from udm_epic5.analysis.convergence import (
    learning_curve,
    stopping_criterion,
)
from udm_epic5.analysis.learning_curve import (
    build_learning_curve_df,
    check_stopping_criterion,
)


# ======================================================================
# MC Dropout (US 5.1)
# ======================================================================


class TestMCDropout:
    """MC Dropout uncertainty estimation correctness."""

    def test_enable_mc_dropout(self):
        """Dropout layers should be in training mode, others in eval."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(10),
            nn.Dropout2d(p=0.3),
        )
        enable_mc_dropout(model)

        # Dropout layers: training mode
        assert model[1].training is True
        assert model[3].training is True
        # Non-dropout layers: eval mode
        assert model[0].training is False
        assert model[2].training is False

    def test_compute_entropy(self):
        """Known probability stack should produce expected entropy values."""
        # Create a prob_stack where mean_prob = 0.5 everywhere -> max entropy
        T, B, C, H, W = 10, 1, 1, 4, 4
        prob_stack = torch.full((T, B, C, H, W), 0.5)
        entropy = compute_entropy(prob_stack)

        # Binary entropy at p=0.5 is ln(2) ~ 0.6931
        expected = np.log(2.0)
        assert torch.allclose(entropy, torch.full_like(entropy, expected), atol=1e-4)

    def test_entropy_shape(self):
        """Output entropy map shape matches [B, 1, H, W]."""
        T, B, C, H, W = 5, 2, 1, 8, 16
        prob_stack = torch.rand(T, B, C, H, W)
        entropy = compute_entropy(prob_stack)
        assert entropy.shape == (B, C, H, W)

    def test_entropy_zero_for_certain(self):
        """When all forward passes agree (prob=1.0), entropy should be ~0."""
        T, B, C, H, W = 10, 1, 1, 4, 4
        prob_stack = torch.ones(T, B, C, H, W)
        entropy = compute_entropy(prob_stack)
        assert entropy.max().item() < 1e-4


# ======================================================================
# Coreset Selection (US 5.2)
# ======================================================================


class TestCoresetSelection:
    """Coreset (farthest-first traversal) selection."""

    def test_budget_respected(self):
        """Returns exactly `budget` indices."""
        features = np.random.randn(100, 16).astype(np.float32)
        indices = coreset_selection(features, budget=10)
        assert len(indices) == 10

    def test_unique_indices(self):
        """No duplicate selections."""
        features = np.random.randn(50, 8).astype(np.float32)
        indices = coreset_selection(features, budget=20)
        assert len(set(indices)) == len(indices)

    def test_diversity(self):
        """Selected points are spread out, not all from one cluster."""
        rng = np.random.RandomState(42)
        # Two distant clusters
        cluster_a = rng.randn(50, 2) + np.array([0, 0])
        cluster_b = rng.randn(50, 2) + np.array([20, 20])
        features = np.vstack([cluster_a, cluster_b]).astype(np.float32)

        indices = coreset_selection(features, budget=10, seed=42)

        # Should pick from both clusters
        from_a = sum(1 for i in indices if i < 50)
        from_b = sum(1 for i in indices if i >= 50)
        assert from_a >= 2, f"Expected at least 2 from cluster A, got {from_a}"
        assert from_b >= 2, f"Expected at least 2 from cluster B, got {from_b}"

    def test_budget_exceeds_pool(self):
        """Budget larger than pool returns all samples."""
        features = np.random.randn(5, 4).astype(np.float32)
        indices = coreset_selection(features, budget=100)
        assert len(indices) == 5


# ======================================================================
# Combined Selection (US 5.2/5.3)
# ======================================================================


class TestCombinedSelection:
    """Combined uncertainty + diversity selection."""

    def test_budget_respected(self):
        """Returns exactly `budget` indices."""
        scores = np.random.rand(100).astype(np.float64)
        features = np.random.randn(100, 16).astype(np.float32)
        indices = combined_selection(scores, features, budget=15)
        assert len(indices) == 15

    def test_alpha_one_is_uncertainty(self):
        """With alpha=1.0, should select the highest-uncertainty samples."""
        n = 50
        scores = np.arange(n, dtype=np.float64)  # 0..49, highest at end
        features = np.random.randn(n, 8).astype(np.float32)
        budget = 5

        indices = combined_selection(scores, features, budget=budget, alpha=1.0)

        # The top-5 uncertainty indices should be 45..49
        top_unc = set(np.argsort(-scores)[:budget])
        selected = set(indices)
        # Allow some overlap tolerance due to greedy iteration
        overlap = len(top_unc & selected)
        assert overlap >= 3, f"Expected most selections from top uncertainty, got {overlap}/5"

    def test_alpha_zero_is_diversity(self):
        """With alpha=0.0, selection should maximise spatial spread."""
        rng = np.random.RandomState(123)
        # Two well-separated clusters
        cluster_a = rng.randn(30, 2) + np.array([0, 0])
        cluster_b = rng.randn(30, 2) + np.array([50, 50])
        features = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        scores = np.ones(60)  # uniform uncertainty

        indices = combined_selection(scores, features, budget=10, alpha=0.0, seed=42)

        from_a = sum(1 for i in indices if i < 30)
        from_b = sum(1 for i in indices if i >= 30)
        assert from_a >= 2, f"Expected cluster A representation, got {from_a}"
        assert from_b >= 2, f"Expected cluster B representation, got {from_b}"

    def test_unique_indices(self):
        """No duplicate selections."""
        scores = np.random.rand(80).astype(np.float64)
        features = np.random.randn(80, 10).astype(np.float32)
        indices = combined_selection(scores, features, budget=25)
        assert len(set(indices)) == len(indices)


# ======================================================================
# Labeling Session (US 5.4)
# ======================================================================


class TestLabelingSession:
    """Labeling session folder management."""

    def test_prepare_creates_folders(self, tmp_path):
        """Session folder structure is created correctly."""
        # Create dummy images and selection CSV
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(5):
            (img_dir / f"img_{i:03d}.png").write_bytes(b"\x89PNG fake data padding")

        csv_path = tmp_path / "selected.csv"
        df = pd.DataFrame({
            "image_path": [f"img_{i:03d}.png" for i in range(5)],
            "mean_entropy": np.random.rand(5),
            "max_entropy": np.random.rand(5),
        })
        df.to_csv(csv_path, index=False)

        session = LabelingSession(
            selected_csv=str(csv_path),
            images_dir=str(img_dir),
            output_dir=str(tmp_path / "labeling"),
            session_name="test_session",
        )
        session.prepare()

        session_dir = tmp_path / "labeling" / "test_session"
        assert (session_dir / "images").is_dir()
        assert (session_dir / "masks").is_dir()
        assert (session_dir / "selected.csv").is_file()

    def test_status_counts(self, tmp_path):
        """Correctly counts labeled vs unlabeled images."""
        # Create a session with images and partial masks
        img_dir = tmp_path / "source_images"
        img_dir.mkdir()
        for i in range(5):
            (img_dir / f"img_{i:03d}.png").write_bytes(b"\x89PNG fake data padding")

        csv_path = tmp_path / "selected.csv"
        df = pd.DataFrame({
            "image_path": [f"img_{i:03d}.png" for i in range(5)],
        })
        df.to_csv(csv_path, index=False)

        session = LabelingSession(
            selected_csv=str(csv_path),
            images_dir=str(img_dir),
            output_dir=str(tmp_path / "labeling"),
            session_name="status_test",
        )
        session.prepare()

        # Simulate labeling 3 out of 5 (write non-empty mask files)
        for i in range(3):
            mask_path = session.masks_out / f"img_{i:03d}.png"
            mask_path.write_bytes(b"\x89PNG real mask content here")

        status = session.status()
        assert status["total"] == 5
        assert status["labeled"] == 3
        assert status["unlabeled"] == 2


# ======================================================================
# Learning Curve (US 5.6)
# ======================================================================


class TestLearningCurve:
    """Learning-curve DataFrame and analysis."""

    def _make_results_csv(self, tmp_path, n_rounds=5):
        """Helper to create a synthetic results CSV."""
        rows = []
        for r in range(1, n_rounds + 1):
            rows.append({
                "round": r,
                "strategy": "combined",
                "budget": 50,
                "dice": 0.5 + 0.05 * r,
                "iou": 0.4 + 0.04 * r,
                "f1": 0.5 + 0.05 * r,
            })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_dataframe_columns(self, tmp_path):
        """Loaded DataFrame has all required columns."""
        csv_path = self._make_results_csv(tmp_path)
        df = build_learning_curve_df(csv_path)
        for col in ["round", "strategy", "budget", "dice", "iou", "f1"]:
            assert col in df.columns

    def test_missing_columns_raises(self, tmp_path):
        """ValueError raised when required columns are missing."""
        df = pd.DataFrame({"round": [1], "strategy": ["x"]})
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)
        try:
            build_learning_curve_df(csv_path)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_stopping_criterion(self, tmp_path):
        """Returns True when improvement is below threshold."""
        csv_path = self._make_results_csv(tmp_path)
        df = build_learning_curve_df(csv_path)
        # Default improvement is 0.05 per round, threshold 0.02 -> should NOT stop
        assert check_stopping_criterion(df, metric="dice", threshold=0.02) is False

    def test_convergence_learning_curve(self):
        """The convergence module learning_curve function builds a tidy DF."""
        results = [
            {"n_labels": 10, "strategy": "uncertainty", "f1": 0.65, "iou": 0.50, "round": 1},
            {"n_labels": 20, "strategy": "uncertainty", "f1": 0.72, "iou": 0.58, "round": 2},
        ]
        df = learning_curve(results)
        assert "n_labels" in df.columns
        assert "strategy" in df.columns
        assert len(df) == 2


# ======================================================================
# Convergence / Stopping (US 5.6)
# ======================================================================


class TestConvergence:
    """Stopping criterion edge cases."""

    def test_stopping_true(self):
        """Small improvement triggers stop."""
        df = pd.DataFrame({
            "round": [1, 2, 3],
            "dice": [0.80, 0.81, 0.811],  # improvement 0.001 < 0.02
        })
        assert check_stopping_criterion(df, metric="dice", threshold=0.02) is True

    def test_stopping_false(self):
        """Large improvement continues training."""
        df = pd.DataFrame({
            "round": [1, 2, 3],
            "dice": [0.60, 0.70, 0.80],  # improvement 0.10 >= 0.02
        })
        assert check_stopping_criterion(df, metric="dice", threshold=0.02) is False

    def test_single_round_no_stop(self):
        """With only one round, stopping should be False."""
        df = pd.DataFrame({"round": [1], "dice": [0.75]})
        assert check_stopping_criterion(df, metric="dice", threshold=0.02) is False

    def test_convergence_module_stopping_true(self):
        """Small F1 improvement triggers stop in convergence module."""
        results = [
            {"n_labels": 10, "strategy": "combined", "f1": 0.80, "iou": 0.70, "round": 1},
            {"n_labels": 20, "strategy": "combined", "f1": 0.805, "iou": 0.71, "round": 2},
        ]
        assert stopping_criterion(results, min_improvement=0.02) is True

    def test_convergence_module_stopping_false(self):
        """Large F1 improvement continues in convergence module."""
        results = [
            {"n_labels": 10, "strategy": "combined", "f1": 0.60, "iou": 0.50, "round": 1},
            {"n_labels": 20, "strategy": "combined", "f1": 0.75, "iou": 0.65, "round": 2},
        ]
        assert stopping_criterion(results, min_improvement=0.02) is False
