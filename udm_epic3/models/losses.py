"""
Epic 3 — CycleGAN loss functions.

Provides the standard CycleGAN losses (LSGAN adversarial, cycle-consistency,
identity) plus a domain-specific defect-preservation loss that penalises
the generator for destroying defect regions during translation.

Usage::

    from udm_epic3.models.losses import (
        adversarial_loss_lsgan,
        cycle_consistency_loss,
        identity_loss,
        defect_preservation_loss,
    )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def adversarial_loss_lsgan(
    pred: torch.Tensor,
    target_is_real: bool,
) -> torch.Tensor:
    """
    Least-squares GAN (LSGAN) adversarial loss.

    Uses MSE with targets of 1.0 (real) or 0.0 (fake) rather than BCE,
    which produces more stable gradients and higher-quality images
    (Mao et al., 2017).

    Args:
        pred:           Raw discriminator output ``[B, 1, H', W']``.
        target_is_real: If ``True`` the target is ``1.0`` (real);
                        otherwise ``0.0`` (fake).

    Returns:
        Scalar MSE loss.
    """
    target_val = 1.0 if target_is_real else 0.0
    target = torch.full_like(pred, target_val)
    return F.mse_loss(pred, target)


def cycle_consistency_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """
    Cycle-consistency loss (L1).

    Ensures that translating an image to the other domain and back
    recovers the original image: ``||G_B2A(G_A2B(x)) - x||_1``.

    Args:
        reconstructed: Reconstructed image after a full translation cycle.
        original:      Original input image.

    Returns:
        Scalar L1 loss.
    """
    return F.l1_loss(reconstructed, original)


def identity_loss(
    identity_output: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """
    Identity regularisation loss (L1).

    When a generator receives an image that already belongs to its target
    domain, it should act as an identity mapping:
    ``||G_A2B(real_B) - real_B||_1``.

    This encourages colour/intensity preservation and stabilises training.

    Args:
        identity_output: Generator output when given a target-domain image.
        original:        The target-domain image itself.

    Returns:
        Scalar L1 loss.
    """
    return F.l1_loss(identity_output, original)


def defect_preservation_loss(
    mask_real: torch.Tensor,
    translated: torch.Tensor,
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Defect-preservation loss based on soft Dice similarity.

    Penalises the generator for destroying or diminishing defect regions
    during cross-modality translation.  The loss compares the binary defect
    mask from the source image against the corresponding intensity region
    in the translated image (thresholded to produce a pseudo-mask).

    A Dice score of 1.0 means perfect overlap, so the *loss* is
    ``1 - Dice`` (lower is better).

    Args:
        mask_real:       Binary defect mask from the source domain,
                         shape ``[B, 1, H, W]`` with values in ``{0, 1}``.
        translated:      Translated image from the generator,
                         shape ``[B, 1, H, W]``.  Values are expected in
                         ``[-1, 1]`` (Tanh output) and will be rescaled
                         internally.
        mask_threshold:  Threshold applied to the normalised translated
                         image to create a pseudo defect mask.

    Returns:
        Scalar Dice loss (``1 - Dice``).
    """
    # Rescale translated image from [-1, 1] to [0, 1]
    translated_norm = (translated + 1.0) / 2.0

    # Create pseudo-mask from translated image
    translated_mask = (translated_norm > mask_threshold).float()

    # Compute soft Dice over the defect regions
    smooth = 1e-6
    intersection = (mask_real * translated_mask).sum()
    union = mask_real.sum() + translated_mask.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice
