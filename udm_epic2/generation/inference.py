"""Epic 2 US 2.4 — generate defect patches with a fine-tuned ControlNet."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from udm_epic2.quality.filter import passes_quality_gate


def generate_defect_samples(
    controlnet_path: Path,
    conditioning_image_paths: List[Path],
    output_dir: Path,
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    prompt: str = "semiconductor x-ray void defect, grayscale, high contrast",
    num_inference_steps: int = 30,
    n_per_conditioning: int = 1,
    seed: Optional[int] = 42,
    quality_filter: bool = True,
    min_laplacian_var: float = 5.0,
) -> List[Path]:
    """
    For each conditioning (edge) image, run SD+ControlNet and save PNGs under ``output_dir``.

    Returns list of saved paths. Requires ``.[epic2]`` extras.
    """
    try:
        import torch
        from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
    except ImportError as e:
        raise ImportError('Install epic2 extras: uv pip install -e ".[epic2]"') from e

    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    saved: List[Path] = []
    idx = 0
    for cpath in conditioning_image_paths:
        cond = Image.open(cpath).convert("RGB")
        for k in range(n_per_conditioning):
            if seed is not None:
                gen.manual_seed(int(seed) + idx)
            out = pipe(
                prompt,
                cond,
                num_inference_steps=num_inference_steps,
                generator=gen,
            ).images[0]
            arr = np.array(out)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if quality_filter:
                ok, _ = passes_quality_gate(bgr, min_laplacian_var=min_laplacian_var)
                if not ok:
                    idx += 1
                    continue
            path = output_dir / f"gen_{idx:05d}.png"
            out.save(path)
            saved.append(path)
            idx += 1

    return saved
