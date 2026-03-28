"""
Epic 8 -- Unified cross-modality pipeline.

Orchestrates synthetic data generation for every configured modality,
dispatching to the correct epic's generator via the
:class:`~udm_epic8.registry.modality_registry.ModalityRegistry`.

Each modality produces images, masks and a per-modality manifest JSON.
A top-level manifest is written at the end that indexes all modalities.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class UnifiedPipelineConfig:
    """Settings for a cross-modality synthetic data generation run.

    Attributes:
        modalities:        List of modality names to generate (e.g. ``['xray', 'aoi']``).
        per_modality_config: Mapping from modality name to its generator-specific
                           configuration dict.  Keys that do not appear in
                           *modalities* are silently ignored.
        output_dir:        Root directory for all generated data.
        total_samples:     Default number of samples per modality (overridden by
                           per-modality ``samples`` key if present).
        train_ratio:       Fraction of samples assigned to train split.
        val_ratio:         Fraction of samples assigned to validation split.
        test_ratio:        Fraction of samples assigned to test split.
    """

    modalities: List[str] = field(default_factory=lambda: ["xray"])
    per_modality_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_dir: str = "data/universal"
    total_samples: int = 1000
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

    # ── helpers ────────────────────────────────────────────────────────────

    def samples_for(self, modality: str) -> int:
        """Return sample count for *modality*, falling back to *total_samples*."""
        cfg = self.per_modality_config.get(modality, {})
        return int(cfg.get("samples", self.total_samples))

    def config_for(self, modality: str) -> dict:
        """Return the generator config dict for *modality* (may be empty)."""
        return dict(self.per_modality_config.get(modality, {}))


# ── Modality -> Epic dispatch map ─────────────────────────────────────────────


_MODALITY_DISPATCH: Dict[str, str] = {
    "xray": "xray",
    "aoi": "aoi_wire",
    "usm": "cyclegan",
    "chromasense": "chromasense",
}


# ── Pipeline ──────────────────────────────────────────────────────────────────


class UnifiedPipeline:
    """Orchestrate synthetic data generation across multiple modalities.

    The pipeline iterates over every enabled modality in ``config.modalities``,
    resolves the corresponding generator via :class:`ModalityRegistry`, and
    writes outputs into per-modality sub-directories under ``config.output_dir``.

    Parameters
    ----------
    config : UnifiedPipelineConfig
        Full cross-modality configuration.
    """

    def __init__(self, config: UnifiedPipelineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> Path:
        """Generate synthetic data for **all** configured modalities.

        Returns the path to the top-level manifest JSON that indexes every
        modality's outputs.
        """
        from udm_epic8.registry.modality_registry import registry

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, Any] = {
            "pipeline": "udm_epic8_universal",
            "modalities": {},
            "splits": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
            },
        }

        for modality in self.config.modalities:
            logger.info("Generating modality: %s", modality)
            t0 = time.time()
            try:
                mod_path = self.run_modality(
                    modality,
                    n_samples=self.config.samples_for(modality),
                )
                elapsed = time.time() - t0
                manifest["modalities"][modality] = {
                    "path": str(mod_path),
                    "samples": self.config.samples_for(modality),
                    "elapsed_s": round(elapsed, 2),
                    "status": "ok",
                }
                logger.info(
                    "Modality %s completed in %.1fs -> %s",
                    modality, elapsed, mod_path,
                )
            except Exception:
                logger.exception("Failed to generate modality: %s", modality)
                manifest["modalities"][modality] = {"status": "error"}

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Manifest written -> %s", manifest_path)
        return manifest_path

    def run_modality(self, modality: str, n_samples: int) -> Path:
        """Generate synthetic data for a **single** modality.

        Dispatches to the correct epic's generator via
        :data:`_MODALITY_DISPATCH` and the :class:`ModalityRegistry`.

        Parameters
        ----------
        modality : str
            Logical modality name (``xray``, ``aoi``, ``usm``,
            ``chromasense``).
        n_samples : int
            Number of synthetic samples to produce.

        Returns
        -------
        Path
            Path to the modality's output directory.
        """
        from udm_epic8.registry.modality_registry import registry

        registry_key = _MODALITY_DISPATCH.get(modality, modality)
        mod_config = self.config.config_for(modality)
        mod_output = str(self.output_dir / modality)

        result_path = registry.generate(
            name=registry_key,
            config=mod_config,
            n_samples=n_samples,
            output_dir=mod_output,
        )
        return result_path

    # ── split helpers ─────────────────────────────────────────────────────

    def compute_split_counts(self, n: int) -> Dict[str, int]:
        """Divide *n* samples into train / val / test counts."""
        n_val = max(1, int(round(n * self.config.val_ratio)))
        n_test = max(1, int(round(n * self.config.test_ratio)))
        n_train = n - n_val - n_test
        return {"train": n_train, "val": n_val, "test": n_test}
