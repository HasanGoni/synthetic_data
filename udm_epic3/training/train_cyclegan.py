"""
Epic 3 — CycleGAN adversarial training loop.

Trains two generator-discriminator pairs (G_A2B / D_B and G_B2A / D_A) for
unpaired cross-modality translation between AOI and USM inspection images.

The training procedure follows Zhu et al. (2017) "Unpaired Image-to-Image
Translation using Cycle-Consistent Adversarial Networks" with LSGAN loss,
cycle-consistency loss, and identity regularisation.  An optional
defect-preservation loss encourages the generators to maintain defect
regions when translating images with known defect masks.

Usage::

    from udm_epic3.training.train_cyclegan import train_cyclegan_from_yaml

    best_ckpt = train_cyclegan_from_yaml("configs/epic3_cyclegan.yaml")
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm

from udm_epic3.data.image_pool import ImagePool
from udm_epic3.data.unpaired_dataset import build_cyclegan_datasets
from udm_epic3.models.discriminator import PatchDiscriminator
from udm_epic3.models.generator import ResnetGenerator

logger = logging.getLogger(__name__)
console = Console()


# ------------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------------


def _linear_lr_lambda(epoch: int, max_epochs: int, decay_start: int) -> float:
    """Return LR multiplier: 1.0 for ``epoch < decay_start``, then linearly
    decay to 0 over the remaining epochs.

    Args:
        epoch: Current epoch (0-indexed).
        max_epochs: Total number of training epochs.
        decay_start: Epoch at which linear decay begins.

    Returns:
        Multiplicative LR factor in ``[0, 1]``.
    """
    if epoch < decay_start:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_start) / (max_epochs - decay_start))


# ------------------------------------------------------------------
# Loss helpers
# ------------------------------------------------------------------


def _lsgan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """Least-squares GAN loss (MSE against 1 for real, 0 for fake).

    Using MSE rather than BCE gives smoother gradients and more stable
    training (Mao et al., 2017).
    """
    target_val = 1.0 if target_is_real else 0.0
    target_tensor = torch.full_like(pred, target_val)
    return nn.functional.mse_loss(pred, target_tensor)


def _cycle_consistency_loss(
    reconstructed: torch.Tensor, original: torch.Tensor,
) -> torch.Tensor:
    """L1 cycle-consistency loss: ``|G_B2A(G_A2B(x)) - x|``."""
    return nn.functional.l1_loss(reconstructed, original)


def _identity_loss(
    identity_output: torch.Tensor, original: torch.Tensor,
) -> torch.Tensor:
    """L1 identity loss: ``|G_A2B(b) - b|`` (generator should be identity
    when given images from the target domain)."""
    return nn.functional.l1_loss(identity_output, original)


def _defect_preservation_loss(
    translated: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss restricted to defect regions defined by *mask*.

    Encourages the generator to preserve defect appearance during translation.
    Regions outside the mask are ignored.

    Args:
        translated: Generated image ``[B, 1, H, W]``.
        original: Source image ``[B, 1, H, W]``.
        mask: Binary defect mask ``[B, 1, H, W]`` (1 = defect).

    Returns:
        Scalar L1 loss over masked pixels (0 if mask is empty).
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=translated.device)
    masked_diff = torch.abs(translated - original) * mask
    return masked_diff.sum() / mask.sum()


# ------------------------------------------------------------------
# Visual sample saving
# ------------------------------------------------------------------


def _save_visual_sample(
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    fake_A: torch.Tensor,
    fake_B: torch.Tensor,
    rec_A: torch.Tensor,
    rec_B: torch.Tensor,
    save_path: Path,
) -> None:
    """Save a horizontal strip of [real_A | fake_B | rec_A | real_B | fake_A | rec_B].

    Takes the first sample from the batch and converts from ``[-1, 1]``
    back to uint8.
    """

    def _to_uint8(t: torch.Tensor) -> np.ndarray:
        img = t[0, 0].detach().cpu().numpy()  # (H, W)
        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return img

    panels = [
        _to_uint8(real_A),
        _to_uint8(fake_B),
        _to_uint8(rec_A),
        _to_uint8(real_B),
        _to_uint8(fake_A),
        _to_uint8(rec_B),
    ]
    strip = np.concatenate(panels, axis=1)
    cv2.imwrite(str(save_path), strip)


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------


def train_cyclegan_from_yaml(config_path: str | Path) -> Path:
    """Train a CycleGAN model from a YAML configuration file.

    The config must contain at least::

        data:
          image_size: [512, 512]
          train:
            dir_A: /path/to/aoi/train
            dir_B: /path/to/usm/train
        training:
          max_epochs: 200
          lr: 0.0002
          lambda_cycle: 10.0
          lambda_idt: 5.0
          lambda_defect: 1.0
          batch_size: 1
          num_workers: 4
          pool_size: 50
          checkpoint_every: 10
          sample_every: 5
          output_dir: outputs/cyclegan
        model:
          n_filters: 64
          n_blocks: 9

    Training procedure per batch:
        1. Forward generators: ``fake_B, rec_A, fake_A, rec_B, idt_A, idt_B``
        2. Generator loss = adversarial + cycle + identity + (optional) defect
        3. Update generators
        4. Discriminator loss with ImagePool for stability
        5. Update discriminators

    LR is held constant for the first half of training and linearly decayed
    to zero in the second half.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Path to the final saved checkpoint.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # --- Hyperparameters ---
    max_epochs: int = train_cfg.get("max_epochs", 200)
    lr: float = train_cfg.get("lr", 2e-4)
    beta1: float = train_cfg.get("beta1", 0.5)
    beta2: float = train_cfg.get("beta2", 0.999)
    lambda_cycle: float = train_cfg.get("lambda_cycle", 10.0)
    lambda_idt: float = train_cfg.get("lambda_idt", 5.0)
    lambda_defect: float = train_cfg.get("lambda_defect", 1.0)
    batch_size: int = train_cfg.get("batch_size", 1)
    num_workers: int = train_cfg.get("num_workers", 4)
    pool_size: int = train_cfg.get("pool_size", 50)
    ckpt_every: int = train_cfg.get("checkpoint_every", 10)
    sample_every: int = train_cfg.get("sample_every", 5)
    decay_start: int = train_cfg.get("decay_start", max_epochs // 2)
    output_dir = Path(train_cfg.get("output_dir", "outputs/cyclegan"))

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Device:[/] {device}")

    # ---- Datasets -----------------------------------------------------------
    datasets = build_cyclegan_datasets(cfg)
    train_ds = datasets["train"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    console.print(
        f"[bold cyan]Training dataset:[/] {len(train_ds)} samples, "
        f"batch_size={batch_size}",
    )

    # ---- Models -------------------------------------------------------------
    in_channels: int = model_cfg.get("in_channels", 1)
    out_channels: int = model_cfg.get("out_channels", 1)
    n_filters: int = model_cfg.get("n_filters", 64)
    n_blocks: int = model_cfg.get("n_blocks", 9)

    G_A2B = ResnetGenerator(
        in_channels=in_channels, out_channels=out_channels,
        n_filters=n_filters, n_blocks=n_blocks,
    ).to(device)
    G_B2A = ResnetGenerator(
        in_channels=in_channels, out_channels=out_channels,
        n_filters=n_filters, n_blocks=n_blocks,
    ).to(device)
    D_A = PatchDiscriminator(in_channels=in_channels, n_filters=n_filters).to(device)
    D_B = PatchDiscriminator(in_channels=in_channels, n_filters=n_filters).to(device)

    console.print(
        f"[bold cyan]Generators:[/] {n_filters}f, {n_blocks} ResNet blocks  "
        f"[bold cyan]Discriminators:[/] PatchGAN {n_filters}f",
    )

    # ---- Optimizers ---------------------------------------------------------
    opt_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=lr, betas=(beta1, beta2),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=lr, betas=(beta1, beta2),
    )

    # ---- LR Schedulers (linear decay in second half) ------------------------
    lr_lambda = lambda epoch: _linear_lr_lambda(epoch, max_epochs, decay_start)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)

    # ---- Image Pools --------------------------------------------------------
    pool_A = ImagePool(pool_size=pool_size)
    pool_B = ImagePool(pool_size=pool_size)

    # ---- Training loop ------------------------------------------------------
    final_ckpt_path = output_dir / "cyclegan_final.pt"

    for epoch in range(1, max_epochs + 1):
        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_cycle = 0.0
        running_loss_idt = 0.0
        running_loss_defect = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
        for batch in pbar:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            has_mask_A = "mask_A" in batch and batch["mask_A"] is not None
            has_mask_B = "mask_B" in batch and batch["mask_B"] is not None
            mask_A: Optional[torch.Tensor] = None
            mask_B: Optional[torch.Tensor] = None
            if has_mask_A:
                mask_A = batch["mask_A"].to(device)
            if has_mask_B:
                mask_B = batch["mask_B"].to(device)

            # ==============================================================
            # 1. Forward generators
            # ==============================================================
            fake_B = G_A2B(real_A)       # A -> B
            rec_A = G_B2A(fake_B)        # A -> B -> A  (cycle)
            fake_A = G_B2A(real_B)       # B -> A
            rec_B = G_A2B(fake_A)        # B -> A -> B  (cycle)
            idt_A = G_B2A(real_A)        # A -> A  (identity)
            idt_B = G_A2B(real_B)        # B -> B  (identity)

            # ==============================================================
            # 2. Generator losses
            # ==============================================================
            # Adversarial losses (generators want discriminators to say "real")
            loss_G_A2B = _lsgan_loss(D_B(fake_B), target_is_real=True)
            loss_G_B2A = _lsgan_loss(D_A(fake_A), target_is_real=True)

            # Cycle-consistency losses
            loss_cycle_A = _cycle_consistency_loss(rec_A, real_A)
            loss_cycle_B = _cycle_consistency_loss(rec_B, real_B)

            # Identity losses
            loss_idt_A = _identity_loss(idt_A, real_A)
            loss_idt_B = _identity_loss(idt_B, real_B)

            # Total generator loss
            loss_G = (
                loss_G_A2B + loss_G_B2A
                + lambda_cycle * (loss_cycle_A + loss_cycle_B)
                + lambda_idt * (loss_idt_A + loss_idt_B)
            )

            # Optional defect-preservation loss
            loss_defect = torch.tensor(0.0, device=device)
            if mask_A is not None:
                loss_defect = loss_defect + _defect_preservation_loss(
                    fake_B, real_A, mask_A,
                )
            if mask_B is not None:
                loss_defect = loss_defect + _defect_preservation_loss(
                    fake_A, real_B, mask_B,
                )
            if mask_A is not None or mask_B is not None:
                loss_G = loss_G + lambda_defect * loss_defect

            # ==============================================================
            # 3. Update generators
            # ==============================================================
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            # ==============================================================
            # 4. Discriminator losses (with ImagePool)
            # ==============================================================
            # Discriminator A: distinguish real_A from fake_A
            fake_A_pool = pool_A.query(fake_A.detach())
            loss_D_A_real = _lsgan_loss(D_A(real_A), target_is_real=True)
            loss_D_A_fake = _lsgan_loss(D_A(fake_A_pool), target_is_real=False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # Discriminator B: distinguish real_B from fake_B
            fake_B_pool = pool_B.query(fake_B.detach())
            loss_D_B_real = _lsgan_loss(D_B(real_B), target_is_real=True)
            loss_D_B_fake = _lsgan_loss(D_B(fake_B_pool), target_is_real=False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B

            # ==============================================================
            # 5. Update discriminators
            # ==============================================================
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # ==============================================================
            # Bookkeeping
            # ==============================================================
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_loss_cycle += (loss_cycle_A.item() + loss_cycle_B.item())
            running_loss_idt += (loss_idt_A.item() + loss_idt_B.item())
            running_loss_defect += loss_defect.item()
            n_batches += 1

            pbar.set_postfix(
                G=f"{loss_G.item():.3f}",
                D=f"{loss_D.item():.3f}",
                cyc=f"{(loss_cycle_A.item() + loss_cycle_B.item()):.3f}",
            )

        # ---- Step LR schedulers --------------------------------------------
        sched_G.step()
        sched_D.step()

        # ---- Epoch logging (rich table) ------------------------------------
        avg_G = running_loss_G / max(n_batches, 1)
        avg_D = running_loss_D / max(n_batches, 1)
        avg_cyc = running_loss_cycle / max(n_batches, 1)
        avg_idt = running_loss_idt / max(n_batches, 1)
        avg_def = running_loss_defect / max(n_batches, 1)
        current_lr = sched_G.get_last_lr()[0]

        table = Table(title=f"Epoch {epoch}/{max_epochs}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("loss_G", f"{avg_G:.4f}")
        table.add_row("loss_D", f"{avg_D:.4f}")
        table.add_row("loss_cycle", f"{avg_cyc:.4f}")
        table.add_row("loss_identity", f"{avg_idt:.4f}")
        table.add_row("loss_defect", f"{avg_def:.4f}")
        table.add_row("lr", f"{current_lr:.6f}")
        console.print(table)

        # ---- Save visual samples --------------------------------------------
        if epoch % sample_every == 0:
            with torch.no_grad():
                G_A2B.eval()
                G_B2A.eval()
                sample_batch = next(iter(train_loader))
                s_real_A = sample_batch["A"][:1].to(device)
                s_real_B = sample_batch["B"][:1].to(device)
                s_fake_B = G_A2B(s_real_A)
                s_rec_A = G_B2A(s_fake_B)
                s_fake_A = G_B2A(s_real_B)
                s_rec_B = G_A2B(s_fake_A)
                _save_visual_sample(
                    s_real_A, s_real_B, s_fake_A, s_fake_B, s_rec_A, s_rec_B,
                    samples_dir / f"epoch_{epoch:04d}.png",
                )
                G_A2B.train()
                G_B2A.train()

        # ---- Checkpointing --------------------------------------------------
        if epoch % ckpt_every == 0:
            ckpt_path = output_dir / f"cyclegan_epoch{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "G_A2B_state_dict": G_A2B.state_dict(),
                    "G_B2A_state_dict": G_B2A.state_dict(),
                    "D_A_state_dict": D_A.state_dict(),
                    "D_B_state_dict": D_B.state_dict(),
                    "opt_G_state_dict": opt_G.state_dict(),
                    "opt_D_state_dict": opt_D.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            console.print(f"[bold yellow]  -> Checkpoint saved: {ckpt_path}[/]")

    # ---- Save final checkpoint -----------------------------------------------
    torch.save(
        {
            "epoch": max_epochs,
            "G_A2B_state_dict": G_A2B.state_dict(),
            "G_B2A_state_dict": G_B2A.state_dict(),
            "D_A_state_dict": D_A.state_dict(),
            "D_B_state_dict": D_B.state_dict(),
            "opt_G_state_dict": opt_G.state_dict(),
            "opt_D_state_dict": opt_D.state_dict(),
            "config": cfg,
        },
        final_ckpt_path,
    )
    console.print(
        f"[bold green]Training complete. Final checkpoint:[/] {final_ckpt_path}",
    )
    return final_ckpt_path
