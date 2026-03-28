"""
Epic 2 US 2.3 — fine-tune ControlNet (canny-style conditioning) on defect crops + edge maps.

Requires optional deps: ``pip install -e ".[epic2]"`` (diffusers, accelerate, transformers).

Based on the Hugging Face ``train_controlnet.py`` loss (noise prediction MSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def train_controlnet_from_yaml(path: str | Path) -> None:
    """Load YAML config and run training on the current CUDA device (or CPU)."""
    path = Path(path)
    with open(path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    try:
        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from diffusers.optimization import get_scheduler
        from torch.utils.data import DataLoader
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "Epic 2 training requires optional deps: uv pip install -e \".[epic2]\""
        ) from e

    from udm_epic2.training.dataset_sd import Epic2ControlNetDataset, collate_fn

    crops_root = Path(cfg["crops_root"])
    output_dir = Path(cfg.get("output_dir", "outputs/epic2_controlnet"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained = cfg.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")
    controlnet_id = cfg.get(
        "controlnet_model_name_or_path",
        "lllyasviel/sd-controlnet-canny",
    )
    resolution = int(cfg.get("resolution", 512))
    caption = str(cfg.get("caption", "semiconductor x-ray void defect"))
    max_steps = int(cfg.get("max_train_steps", 500))
    batch_size = int(cfg.get("train_batch_size", 1))
    lr = float(cfg.get("learning_rate", 1e-5))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    save_every = int(cfg.get("checkpointing_steps", 250))

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=cfg.get("mixed_precision", "fp16"),
    )

    tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(controlnet_id)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    train_ds = Epic2ControlNetDataset(
        crops_root,
        resolution=resolution,
        caption=caption,
        tokenizer=tokenizer,
    )
    if len(train_ds) == 0:
        raise ValueError(f"No samples in {crops_root} — run extract-crops with --write-edges first.")

    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(cfg.get("dataloader_num_workers", 0)),
    )

    opt = torch.optim.AdamW(controlnet.parameters(), lr=lr)
    lr_sched = get_scheduler(
        cfg.get("lr_scheduler", "constant"),
        optimizer=opt,
        num_warmup_steps=int(cfg.get("lr_warmup_steps", 0)),
        num_training_steps=max_steps,
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    controlnet, opt, loader, lr_sched = accelerator.prepare(controlnet, opt, loader, lr_sched)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    epoch = 0
    while global_step < max_steps:
        for batch in loader:
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents.float(), noise.float(), timesteps
                ).to(dtype=weight_dtype)

                enc = text_encoder(batch["input_ids"], return_dict=False)[0]
                cond = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down, mid = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=enc,
                    controlnet_cond=cond,
                    return_dict=False,
                )

                pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=enc,
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down],
                    mid_block_additional_residual=mid.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                target = noise
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
                    opt.step()
                    lr_sched.step()
                    opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % save_every == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(controlnet)
                    unwrapped.save_pretrained(save_path)

                if global_step >= max_steps:
                    break
        epoch += 1
        if global_step >= max_steps:
            break

    if accelerator.is_main_process:
        final = output_dir / "controlnet_final"
        final.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(final)
