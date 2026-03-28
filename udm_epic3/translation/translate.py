"""
Epic 3 — CycleGAN inference / translation utilities.

Provides functions to translate single images or entire directories between
AOI and USM modalities using a trained CycleGAN checkpoint.

Usage::

    from udm_epic3.translation.translate import translate_single, translate_directory

    # Single image
    result = translate_single(model, "input.png", direction="a2b")

    # Entire directory
    translate_directory(
        model_path="outputs/cyclegan/cyclegan_final.pt",
        input_dir="data/aoi/test",
        output_dir="outputs/translated_usm",
        direction="a2b",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from udm_epic3.models.generator import ResnetGenerator

logger = logging.getLogger(__name__)

# Supported image file extensions (case-insensitive matching).
_IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".bmp"}


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------


def _load_and_preprocess(
    image_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load a grayscale image, resize, normalise to ``[-1, 1]``, return tensor.

    Returns:
        Tuple of (batch tensor ``[1, 1, H, W]``, original ``(H, W)``).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    original_size = (img.shape[0], img.shape[1])

    if (img.shape[0], img.shape[1]) != image_size:
        img = cv2.resize(
            img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR,
        )

    # Normalise to [-1, 1] (Tanh range)
    img_f = img.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(device), original_size


def _postprocess(tensor: torch.Tensor, target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Convert generator output ``[1, 1, H, W]`` in ``[-1, 1]`` to uint8 HxW.

    If *target_size* is given, resize back to the original image dimensions.
    """
    img = tensor[0, 0].detach().cpu().numpy()
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    if target_size is not None and (img.shape[0], img.shape[1]) != target_size:
        img = cv2.resize(
            img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR,
        )
    return img


def _load_generator_from_checkpoint(
    model_path: Path,
    direction: str,
    device: torch.device,
    config: Optional[dict] = None,
) -> tuple[ResnetGenerator, tuple[int, int]]:
    """Load a generator from a CycleGAN checkpoint.

    Args:
        model_path: Path to the ``.pt`` checkpoint file.
        direction: ``"a2b"`` to load G_A2B, ``"b2a"`` to load G_B2A.
        device: Target device.
        config: Optional override config; if None, uses config from checkpoint.

    Returns:
        Tuple of (generator model, image_size).
    """
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    cfg = config if config is not None else checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    in_channels: int = model_cfg.get("in_channels", 1)
    out_channels: int = model_cfg.get("out_channels", 1)
    n_filters: int = model_cfg.get("n_filters", 64)
    n_blocks: int = model_cfg.get("n_blocks", 9)
    image_size = tuple(data_cfg.get("image_size", [512, 512]))

    generator = ResnetGenerator(
        in_channels=in_channels, out_channels=out_channels,
        n_filters=n_filters, n_blocks=n_blocks,
    ).to(device)

    state_key = "G_A2B_state_dict" if direction == "a2b" else "G_B2A_state_dict"
    if state_key not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain '{state_key}'. "
            f"Available keys: {list(checkpoint.keys())}",
        )
    generator.load_state_dict(checkpoint[state_key])
    generator.eval()

    logger.info(
        "Loaded generator '%s' from %s  (%d params)",
        state_key, model_path,
        sum(p.numel() for p in generator.parameters()),
    )
    return generator, image_size


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def translate_single(
    model: ResnetGenerator,
    image_path: Union[str, Path],
    direction: str = "a2b",
    device: str = "cuda",
    image_size: tuple[int, int] = (512, 512),
    resize_back: bool = True,
) -> np.ndarray:
    """Translate a single image through a trained generator.

    Args:
        model: A :class:`ResnetGenerator` instance (already loaded and on device).
        image_path: Path to the input image file.
        direction: ``"a2b"`` or ``"b2a"`` (informational; the caller must
            supply the correct generator).
        device: Device string (``"cuda"`` or ``"cpu"``).
        image_size: ``(height, width)`` the model expects.
        resize_back: If True, resize the output back to the original
            image dimensions.

    Returns:
        Translated image as a uint8 numpy array of shape ``(H, W)``.
    """
    dev = torch.device(device)
    image_path = Path(image_path)

    input_tensor, original_size = _load_and_preprocess(image_path, image_size, dev)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    target_size = original_size if resize_back else None
    return _postprocess(output_tensor, target_size=target_size)


def translate_directory(
    model_path: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    direction: str = "a2b",
    device: str = "cuda",
    config: Optional[dict] = None,
) -> None:
    """Translate all images in a directory using a trained CycleGAN checkpoint.

    Loads the appropriate generator from the checkpoint, processes every
    supported image file in *input_dir*, and saves the results to
    *output_dir* with the same filenames.

    Args:
        model_path: Path to the CycleGAN ``.pt`` checkpoint.
        input_dir: Directory containing input images.
        output_dir: Directory to save translated images.
        direction: ``"a2b"`` (A->B) or ``"b2a"`` (B->A).
        device: Device string (``"cuda"`` or ``"cpu"``).
        config: Optional override config dict; if None, uses the config
            stored in the checkpoint.
    """
    model_path = Path(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # Load generator
    generator, image_size = _load_generator_from_checkpoint(
        model_path, direction, dev, config=config,
    )

    # Collect input images
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not image_paths:
        logger.warning("No images found in %s", input_dir)
        return

    logger.info(
        "Translating %d images  direction=%s  device=%s",
        len(image_paths), direction, device,
    )

    from tqdm import tqdm

    for img_path in tqdm(image_paths, desc=f"Translating ({direction})"):
        result = translate_single(
            model=generator,
            image_path=img_path,
            direction=direction,
            device=device,
            image_size=image_size,
            resize_back=True,
        )
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), result)

    logger.info("Saved %d translated images to %s", len(image_paths), output_dir)
