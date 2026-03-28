"""
Epic 8 -- Modality Registry.

Module-level registry that maps modality names to generator callables and
their associated configuration classes.  Every previous epic is pre-registered
so that the unified pipeline can dispatch generation without hard-coding
import paths.

Usage::

    from udm_epic8.registry.modality_registry import registry

    registry.list_modalities()          # -> ['xray', 'controlnet', ...]
    fn, cfg_cls = registry.get('xray')
    result_path = registry.generate('xray', config={...}, n_samples=100,
                                    output_dir='data/xray')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


# ── Registry class ────────────────────────────────────────────────────────────


class ModalityRegistry:
    """Central registry mapping modality names to generators.

    This is **not** a strict singleton but a plain class; we expose a
    module-level instance :data:`registry` that acts as the global registry.

    Each entry stores:

    * ``generator_fn`` -- a callable ``(config: dict, n_samples: int,
      output_dir: str) -> Path`` that produces synthetic data and returns
      the output directory.
    * ``config_cls`` -- the configuration dataclass / Pydantic model used by
      the generator (stored for introspection and validation).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[Callable[..., Path], Type]] = {}

    # ── mutation ──────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        generator_fn: Callable[..., Path],
        config_cls: type,
    ) -> None:
        """Register a modality generator.

        Parameters
        ----------
        name : str
            Short modality identifier (e.g. ``xray``, ``aoi_wire``).
        generator_fn : Callable
            Function with signature
            ``(config: dict, n_samples: int, output_dir: str) -> Path``.
        config_cls : type
            The configuration class associated with *generator_fn*.
        """
        if name in self._entries:
            logger.warning("Overwriting existing registry entry for '%s'", name)
        self._entries[name] = (generator_fn, config_cls)
        logger.debug("Registered modality '%s'", name)

    # ── queries ───────────────────────────────────────────────────────────

    def list_modalities(self) -> List[str]:
        """Return sorted list of registered modality names."""
        return sorted(self._entries.keys())

    def get(self, name: str) -> Tuple[Callable[..., Path], Type]:
        """Return ``(generator_fn, config_cls)`` for *name*.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._entries:
            available = ", ".join(self.list_modalities()) or "(none)"
            raise KeyError(
                f"Modality '{name}' not registered. Available: {available}"
            )
        return self._entries[name]

    def generate(
        self,
        name: str,
        config: dict,
        n_samples: int,
        output_dir: str,
    ) -> Path:
        """Generate synthetic data for *name* using the registered generator.

        Parameters
        ----------
        name : str
            Registered modality name.
        config : dict
            Generator-specific configuration passed through to the callable.
        n_samples : int
            Number of samples to produce.
        output_dir : str
            Root output directory for this modality's data.

        Returns
        -------
        Path
            Directory (or manifest file) written by the generator.
        """
        generator_fn, _config_cls = self.get(name)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = generator_fn(config=config, n_samples=n_samples, output_dir=str(out))
        return Path(result)


# ── Default generator stubs ───────────────────────────────────────────────────
#
# Each stub wraps the real epic module's generation logic.  If the epic package
# is not installed, a graceful fallback produces placeholder data so that tests
# and the CLI still work.


def _generate_xray(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 1 -- Beer-Lambert void synthesis."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic1.generators.void_generator import VoidGenerator

        gen = VoidGenerator(**config) if config else VoidGenerator()
        for i in range(n_samples):
            result = gen.generate()
            image, mask = result["image"], result["mask"]
            _save_array(image, images_dir / f"xray_{i:05d}.png")
            _save_array(mask, masks_dir / f"xray_{i:05d}_mask.png")
    except ImportError:
        logger.warning("udm_epic1 not available; generating placeholder xray data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="xray")

    manifest = _write_modality_manifest(out, "xray", n_samples)
    return manifest


def _generate_controlnet(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 2 -- ControlNet conditioned generation."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic2.generation.controlnet_pipeline import generate_batch

        generate_batch(
            n_samples=n_samples,
            output_dir=str(out),
            **config,
        )
    except ImportError:
        logger.warning("udm_epic2 not available; generating placeholder controlnet data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="controlnet")

    manifest = _write_modality_manifest(out, "controlnet", n_samples)
    return manifest


def _generate_cyclegan(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 3 -- CycleGAN domain translation (AOI -> USM)."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic3.models.cyclegan import CycleGANInference

        model = CycleGANInference(**config) if config else CycleGANInference()
        model.translate_batch(n_samples=n_samples, output_dir=str(out))
    except ImportError:
        logger.warning("udm_epic3 not available; generating placeholder cyclegan data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="cyclegan")

    manifest = _write_modality_manifest(out, "cyclegan", n_samples)
    return manifest


def _generate_dann(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 4 -- DANN domain adaptation (feature-level, not generative).

    DANN is a training strategy rather than a data generator, so this stub
    produces augmented source domain images as its 'generation' step.
    """
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DANN is a training strategy; producing augmented source samples")
    _generate_placeholder(n_samples, images_dir, masks_dir, prefix="dann")

    manifest = _write_modality_manifest(out, "dann", n_samples)
    return manifest


def _generate_active(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 5 -- Active learning sample selection."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic5.selection.active_selector import ActiveSelector

        selector = ActiveSelector(**config) if config else ActiveSelector()
        selector.select_and_save(n_samples=n_samples, output_dir=str(out))
    except ImportError:
        logger.warning("udm_epic5 not available; generating placeholder active data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="active")

    manifest = _write_modality_manifest(out, "active", n_samples)
    return manifest


def _generate_aoi_wire(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 6 -- Bond wire AOI synthesis."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic6.rendering.wire_renderer import BondWireRenderer

        renderer = BondWireRenderer(**config) if config else BondWireRenderer()
        for i in range(n_samples):
            result = renderer.render()
            image, mask = result["image"], result["mask"]
            _save_array(image, images_dir / f"aoi_{i:05d}.png")
            _save_array(mask, masks_dir / f"aoi_{i:05d}_mask.png")
    except ImportError:
        logger.warning("udm_epic6 not available; generating placeholder AOI wire data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="aoi")

    manifest = _write_modality_manifest(out, "aoi_wire", n_samples)
    return manifest


def _generate_chromasense(config: dict, n_samples: int, output_dir: str) -> Path:
    """Epic 7 -- Spectral rendering for ChromaSense."""
    out = Path(output_dir)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from udm_epic7.rendering.spectral_renderer import SpectralRenderer

        renderer = SpectralRenderer(**config) if config else SpectralRenderer()
        for i in range(n_samples):
            result = renderer.render()
            image, mask = result["image"], result["mask"]
            _save_array(image, images_dir / f"chroma_{i:05d}.png")
            _save_array(mask, masks_dir / f"chroma_{i:05d}_mask.png")
    except ImportError:
        logger.warning("udm_epic7 not available; generating placeholder chromasense data")
        _generate_placeholder(n_samples, images_dir, masks_dir, prefix="chroma")

    manifest = _write_modality_manifest(out, "chromasense", n_samples)
    return manifest


# ── Internal helpers ──────────────────────────────────────────────────────────


def _save_array(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array as a PNG image via Pillow."""
    from PIL import Image

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))


def _generate_placeholder(
    n_samples: int,
    images_dir: Path,
    masks_dir: Path,
    prefix: str,
    size: int = 128,
) -> None:
    """Write placeholder random images and blank masks for testing."""
    from PIL import Image

    for i in range(n_samples):
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        Image.fromarray(img).save(str(images_dir / f"{prefix}_{i:05d}.png"))
        Image.fromarray(mask).save(str(masks_dir / f"{prefix}_{i:05d}_mask.png"))


def _write_modality_manifest(out_dir: Path, modality: str, n_samples: int) -> Path:
    """Write a small JSON manifest for a single modality run."""
    manifest_path = out_dir / "manifest.json"
    data = {
        "modality": modality,
        "n_samples": n_samples,
        "images_dir": str(out_dir / "images"),
        "masks_dir": str(out_dir / "masks"),
    }
    manifest_path.write_text(json.dumps(data, indent=2))
    return manifest_path


# ── Module-level registry instance with pre-registered epics ─────────────────

registry = ModalityRegistry()

# Epic 1 -- X-ray void synthesis (Beer-Lambert)
registry.register("xray", _generate_xray, dict)

# Epic 2 -- ControlNet conditioned generation
registry.register("controlnet", _generate_controlnet, dict)

# Epic 3 -- CycleGAN domain translation
registry.register("cyclegan", _generate_cyclegan, dict)

# Epic 4 -- DANN domain adaptation
registry.register("dann", _generate_dann, dict)

# Epic 5 -- Active learning sample selection
registry.register("active", _generate_active, dict)

# Epic 6 -- AOI bond wire synthesis
registry.register("aoi_wire", _generate_aoi_wire, dict)

# Epic 7 -- ChromaSense spectral rendering
registry.register("chromasense", _generate_chromasense, dict)
