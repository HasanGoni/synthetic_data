"""
Epic 3 — Image-quality and defect-preservation metrics for CycleGAN evaluation.

Provides structural similarity (SSIM), defect Dice coefficient, and a
convenience function to evaluate a full directory of translated images
against real references.

Usage::

    from udm_epic3.evaluation.quality_metrics import (
        compute_ssim, compute_defect_dice, evaluate_translation,
    )

    ssim = compute_ssim(real_img, translated_img)
    dice = compute_defect_dice(mask_real, mask_translated)
    df   = evaluate_translation("data/real", "data/translated", masks_dir="data/masks")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ── Per-image metrics ─────────────────────────────────────────────────────────


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    Images should be 2-D ``(H, W)`` or 3-D ``(H, W, C)`` arrays with the
    same shape and dtype.  Internally uses
    :func:`skimage.metrics.structural_similarity` (lazy-imported to avoid
    hard dependency on scikit-image at module load time).

    Args:
        img_a: Reference image.
        img_b: Comparison image.

    Returns:
        SSIM value in ``[-1, 1]`` (higher is better; 1.0 = identical).
    """
    from skimage.metrics import structural_similarity

    # Determine data range from dtype
    if img_a.dtype == np.uint8:
        data_range = 255.0
    else:
        data_range = float(img_a.max() - img_a.min()) if img_a.max() != img_a.min() else 1.0

    channel_axis: Optional[int] = None
    if img_a.ndim == 3 and img_a.shape[-1] in (1, 3, 4):
        channel_axis = -1

    return float(
        structural_similarity(
            img_a, img_b,
            data_range=data_range,
            channel_axis=channel_axis,
        )
    )


def compute_defect_dice(mask_real: np.ndarray, mask_translated: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    .. math::
        \\text{Dice} = \\frac{2 \\cdot |A \\cap B|}{|A| + |B|}

    Both masks are binarised (> 0.5) before comparison.  If both masks are
    entirely empty the Dice is defined as 1.0 (perfect agreement on
    absence of defects).

    Args:
        mask_real:       Ground-truth binary defect mask.
        mask_translated: Predicted / translated defect mask.

    Returns:
        Dice coefficient in ``[0, 1]``.
    """
    a = (mask_real > 0.5).astype(np.float64).ravel()
    b = (mask_translated > 0.5).astype(np.float64).ravel()

    sum_a = a.sum()
    sum_b = b.sum()

    if sum_a + sum_b == 0.0:
        return 1.0  # both empty — perfect agreement

    intersection = (a * b).sum()
    return float(2.0 * intersection / (sum_a + sum_b))


# ── FID (Fréchet Inception Distance) ─────────────────────────────────────────


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Compute Fréchet Inception Distance between two image directories.

    Uses InceptionV3 features (pool3 layer, 2048-d) to compute the
    distance between the real and fake image distributions.  Both
    directories must contain image files with common extensions.

    .. note::
        Requires ``torch`` and ``torchvision``.

    Args:
        real_dir:   Directory of real images.
        fake_dir:   Directory of generated / translated images.
        batch_size: Batch size for feature extraction.
        device:     Torch device string.

    Returns:
        FID score (lower is better; 0.0 = identical distributions).
    """
    import torch
    import torchvision.transforms as T
    from torchvision.models import inception_v3

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)

    # Feature extraction model (InceptionV3 up to pool3)
    model = inception_v3(weights=None, init_weights=False, transform_input=False)
    # Replace final FC with identity to get 2048-d features
    model.fc = torch.nn.Identity()
    model.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _extract_features(img_dir: Path) -> np.ndarray:
        import cv2

        paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        features_list: list[np.ndarray] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                tensors.append(transform(img))
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors).to(device)
            with torch.no_grad():
                feats = model(batch_tensor)
            features_list.append(feats.cpu().numpy())

        return np.concatenate(features_list, axis=0) if features_list else np.empty((0, 2048))

    feats_real = _extract_features(real_dir)
    feats_fake = _extract_features(fake_dir)

    if feats_real.shape[0] < 2 or feats_fake.shape[0] < 2:
        logger.warning("Too few images for reliable FID computation.")
        return float("inf")

    # Compute statistics
    mu_real = np.mean(feats_real, axis=0)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real @ sigma_fake))
    from scipy.linalg import sqrtm

    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    # Numerical stability — discard imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean))
    return max(fid, 0.0)


# ── Directory-level evaluation ────────────────────────────────────────────────


def evaluate_translation(
    real_dir: str | Path,
    translated_dir: str | Path,
    masks_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate translated images against real references.

    For each image found in *translated_dir* whose filename matches a file
    in *real_dir*, compute SSIM.  If *masks_dir* is provided and a matching
    mask file exists, also compute the defect Dice coefficient.

    Args:
        real_dir:       Directory of real reference images.
        translated_dir: Directory of translated (generated) images.
        masks_dir:      Optional directory of binary defect masks (matched
                        by filename).

    Returns:
        :class:`pandas.DataFrame` with columns ``filename``, ``ssim``,
        and (optionally) ``dice``.
    """
    import cv2

    real_dir = Path(real_dir)
    translated_dir = Path(translated_dir)
    masks_dir = Path(masks_dir) if masks_dir else None

    # Build lookup of real images by filename
    real_files = {p.name: p for p in real_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS}

    rows: list[dict] = []
    for tp in sorted(translated_dir.iterdir()):
        if tp.suffix.lower() not in _IMAGE_EXTS:
            continue
        if tp.name not in real_files:
            logger.debug("No matching real image for %s — skipping.", tp.name)
            continue

        real_img = cv2.imread(str(real_files[tp.name]), cv2.IMREAD_GRAYSCALE)
        trans_img = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        if real_img is None or trans_img is None:
            continue

        # Resize translated to match real if needed
        if real_img.shape != trans_img.shape:
            trans_img = cv2.resize(
                trans_img, (real_img.shape[1], real_img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        ssim_val = compute_ssim(real_img, trans_img)
        row: dict = {"filename": tp.name, "ssim": ssim_val}

        if masks_dir is not None:
            mask_path = masks_dir / tp.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Create pseudo-mask from translated image using Otsu threshold
                    _, trans_mask = cv2.threshold(
                        trans_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )
                    if mask.shape != trans_mask.shape:
                        trans_mask = cv2.resize(
                            trans_mask, (mask.shape[1], mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    row["dice"] = compute_defect_dice(mask, trans_mask)

        rows.append(row)

    if not rows:
        logger.warning("No matching image pairs found for evaluation.")
        return pd.DataFrame(columns=["filename", "ssim", "dice"])

    df = pd.DataFrame(rows)
    logger.info(
        "Evaluated %d image pairs — mean SSIM=%.4f%s",
        len(df),
        df["ssim"].mean(),
        f", mean Dice={df['dice'].mean():.4f}" if "dice" in df.columns else "",
    )
    return df
