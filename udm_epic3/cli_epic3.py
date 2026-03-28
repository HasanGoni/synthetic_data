"""
UDM Epic 3 CLI — CycleGAN cross-modality translation: prepare, train, translate, evaluate,
downstream, validate.

Examples
--------
udm-epic3 prepare    --config configs/epic3_cyclegan.yaml
udm-epic3 train      --config configs/epic3_cyclegan.yaml
udm-epic3 translate  --checkpoint outputs/epic3_cyclegan/cyclegan_final.pt --input data/aoi/test --output outputs/translated --direction a2b
udm-epic3 evaluate   --real-dir data/usm/test --translated-dir outputs/translated --masks-dir data/usm/masks
udm-epic3 downstream --translated-dir outputs/translated --masks-dir data/usm/masks
udm-epic3 validate   --checkpoint outputs/epic3_cyclegan/cyclegan_final.pt --config configs/epic3_cyclegan.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic3",
    help="UDM Epic 3 — CycleGAN Cross-Modality Translation (AOI <-> USM)",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── US 3.1 — Prepare: analyse multi-modal data ──────────────────────────────


@app.command("prepare")
def prepare(
    config: Path = typer.Option(
        "configs/epic3_cyclegan.yaml", "--config", "-c",
        help="CycleGAN config YAML with domain paths",
    ),
):
    """US 3.1 -- Analyse multi-modal data: compute stats per modality."""
    import numpy as np
    from rich.table import Table

    cfg = _load_yaml(config)
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)
    data_cfg = cfg.get("data", {})

    table = Table(title="Multi-Modal Dataset Summary")
    table.add_column("Domain", style="cyan")
    table.add_column("Split", style="green")
    table.add_column("Images", justify="right")
    table.add_column("Mean Intensity", justify="right")
    table.add_column("Std Intensity", justify="right")

    for domain_key in ["domain_a", "domain_b"]:
        domain = data_cfg.get(domain_key, {})
        name = domain.get("name", domain_key)

        for split_key in ["train", "val"]:
            split_dir = domain.get(split_key)
            if not split_dir:
                continue
            split_path = Path(split_dir)
            if not split_path.is_dir():
                table.add_row(name, split_key, "[red]NOT FOUND[/]", "-", "-")
                continue

            import cv2

            image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            image_files = sorted(
                p for p in split_path.iterdir() if p.suffix.lower() in image_exts
            )
            n_images = len(image_files)
            if n_images == 0:
                table.add_row(name, split_key, "0", "-", "-")
                continue

            # Sample up to 50 images for intensity statistics
            sample_files = image_files[:min(50, n_images)]
            means = []
            for img_path in sample_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    means.append(float(img.mean()))

            if means:
                mean_val = np.mean(means)
                std_val = np.std(means)
                table.add_row(
                    name, split_key, str(n_images),
                    f"{mean_val:.2f}", f"{std_val:.2f}",
                )
            else:
                table.add_row(name, split_key, str(n_images), "-", "-")

    # Check for paired masks
    paired_masks = data_cfg.get("paired_masks", {}) or {}
    if paired_masks:
        mask_table = Table(title="Defect Masks")
        mask_table.add_column("Domain", style="cyan")
        mask_table.add_column("Path")
        mask_table.add_column("Files", justify="right")
        for key in ["masks_a", "masks_b"]:
            mask_dir = paired_masks.get(key)
            if mask_dir:
                mp = Path(mask_dir)
                if mp.is_dir():
                    n = sum(1 for p in mp.iterdir() if p.is_file())
                    mask_table.add_row(key, str(mp), str(n))
                else:
                    mask_table.add_row(key, str(mp), "[red]NOT FOUND[/]")
        rprint(mask_table)

    rprint(table)
    rprint("[bold green]Done.[/] Review the table above to verify modality distributions.")


# ── US 3.2/3.3 — Train CycleGAN ─────────────────────────────────────────────


@app.command("train")
def train(
    config: Path = typer.Option(
        "configs/epic3_cyclegan.yaml", "--config", "-c",
        help="CycleGAN training config YAML",
    ),
):
    """US 3.2/3.3 -- Train CycleGAN (baseline or defect-aware).

    Set lambda_defect=0 in config for baseline CycleGAN (US 3.2).
    Set lambda_defect>0 for defect-aware training (US 3.3).
    """
    from udm_epic3.training.train_cyclegan import train_cyclegan_from_yaml

    rprint("[bold cyan]Starting CycleGAN training...[/]")
    final_ckpt = train_cyclegan_from_yaml(config)
    rprint(f"[bold green]Training complete.[/] Final checkpoint: [cyan]{final_ckpt}[/]")


# ── US 3.4 — Translate images ───────────────────────────────────────────────


@app.command("translate")
def translate(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="CycleGAN checkpoint (.pt)",
    ),
    input_dir: Path = typer.Option(
        ..., "--input", "-i", help="Directory of source images to translate",
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Directory for translated output images",
    ),
    direction: str = typer.Option(
        "a2b", "--direction", "-d",
        help="Translation direction: 'a2b' (AOI->USM) or 'b2a' (USM->AOI)",
    ),
    device: str = typer.Option(
        "auto", "--device", help="Torch device ('cuda', 'cpu', or 'auto')",
    ),
):
    """US 3.4 -- Translate images from one modality to another."""
    import torch

    from udm_epic3.translation.translate import translate_directory

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rprint(
        f"[bold cyan]Translating[/] {input_dir} -> {output_dir}  "
        f"direction={direction}  device={device}"
    )

    translate_directory(
        model_path=checkpoint,
        input_dir=input_dir,
        output_dir=output_dir,
        direction=direction,
        device=device,
    )
    rprint(f"[bold green]Done.[/] Translated images saved to [cyan]{output_dir}[/]")


# ── US 3.4 — Evaluate translation quality ───────────────────────────────────


@app.command("evaluate")
def evaluate(
    real_dir: Path = typer.Option(
        ..., "--real-dir", "-r", help="Directory of real reference images",
    ),
    translated_dir: Path = typer.Option(
        ..., "--translated-dir", "-t", help="Directory of translated images",
    ),
    masks_dir: Optional[Path] = typer.Option(
        None, "--masks-dir", "-m", help="Optional directory of defect masks (Dice)",
    ),
    output_csv: Optional[Path] = typer.Option(
        None, "--output-csv", "-o", help="Path to save results CSV",
    ),
    compute_fid_flag: bool = typer.Option(
        False, "--fid", help="Also compute FID (requires sufficient images)",
    ),
):
    """US 3.4 -- Compute SSIM, FID, and Dice on translated images."""
    from udm_epic3.evaluation.quality_metrics import (
        compute_fid as _compute_fid,
        evaluate_translation,
    )
    from rich.table import Table

    rprint("[bold cyan]Evaluating translation quality...[/]")

    df = evaluate_translation(
        real_dir=str(real_dir),
        translated_dir=str(translated_dir),
        masks_dir=str(masks_dir) if masks_dir else None,
    )

    if df.empty:
        rprint("[bold red]No matching image pairs found for evaluation.[/]")
        raise typer.Exit(1)

    # Summary table
    table = Table(title="Translation Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    table.add_row(
        "SSIM",
        f"{df['ssim'].mean():.4f}",
        f"{df['ssim'].std():.4f}",
        f"{df['ssim'].min():.4f}",
        f"{df['ssim'].max():.4f}",
    )

    if "dice" in df.columns:
        dice_vals = df["dice"].dropna()
        if len(dice_vals) > 0:
            table.add_row(
                "Dice",
                f"{dice_vals.mean():.4f}",
                f"{dice_vals.std():.4f}",
                f"{dice_vals.min():.4f}",
                f"{dice_vals.max():.4f}",
            )

    rprint(table)
    rprint(f"Evaluated {len(df)} image pairs.")

    # Optional FID
    if compute_fid_flag:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rprint("[bold cyan]Computing FID...[/]")
        fid_val = _compute_fid(str(real_dir), str(translated_dir), device=device)
        rprint(f"[bold green]FID:[/] {fid_val:.2f}")

    # Save CSV
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        rprint(f"Results saved to [cyan]{output_csv}[/]")


# ── US 3.5 — Downstream segmentation ────────────────────────────────────────


@app.command("downstream")
def downstream(
    translated_dir: Path = typer.Option(
        ..., "--translated-dir", "-t",
        help="Directory of translated training images",
    ),
    masks_dir: Path = typer.Option(
        ..., "--masks-dir", "-m",
        help="Directory of defect masks for segmentation training",
    ),
    model_type: str = typer.Option(
        "unet", "--model", help="Segmentation model type (unet, deeplabv3)",
    ),
    epochs: int = typer.Option(50, "--epochs", help="Number of training epochs"),
    output_dir: Path = typer.Option(
        "outputs/epic3_downstream", "--output", "-o",
        help="Output directory for trained segmentation model",
    ),
):
    """US 3.5 -- Train segmentation model on translated data (placeholder).

    This command provides instructions for training a downstream
    segmentation model using the CycleGAN-translated images as augmented
    training data.
    """
    from rich.panel import Panel

    instructions = (
        f"[bold]Downstream Segmentation Training[/]\n\n"
        f"1. Use translated images from: [cyan]{translated_dir}[/]\n"
        f"2. Pair with defect masks from: [cyan]{masks_dir}[/]\n"
        f"3. Model architecture: [green]{model_type}[/]\n"
        f"4. Training epochs: [green]{epochs}[/]\n"
        f"5. Output directory: [cyan]{output_dir}[/]\n\n"
        f"[bold yellow]Integration with Epic 4 (DANN):[/]\n"
        f"   The translated images can serve as additional training data\n"
        f"   for the DANN domain adaptation pipeline:\n\n"
        f"   [dim]udm-epic4 train --config configs/epic4_dann.yaml[/]\n\n"
        f"   Add the translated directory as an extra target domain in\n"
        f"   the Epic 4 config to improve cross-site generalisation.\n\n"
        f"[bold yellow]Standalone segmentation training:[/]\n"
        f"   Use the translated images with any segmentation framework.\n"
        f"   Example with segmentation-models-pytorch:\n\n"
        f"   [dim]import segmentation_models_pytorch as smp\n"
        f"   model = smp.Unet('resnet34', in_channels=1, classes=1)[/]"
    )

    rprint(Panel(instructions, title="US 3.5 — Downstream Task", border_style="green"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rprint(f"\n[bold green]Output directory created:[/] [cyan]{output_dir}[/]")


# ── US 3.6 — Multi-site validation ──────────────────────────────────────────


@app.command("validate")
def validate(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="CycleGAN checkpoint (.pt)",
    ),
    config: Path = typer.Option(
        "configs/epic3_cyclegan.yaml", "--config", "-c",
        help="Config with data paths",
    ),
    site_dirs: Optional[list[str]] = typer.Option(
        None, "--site", "-s",
        help="Additional site directories to evaluate (repeatable)",
    ),
    direction: str = typer.Option(
        "a2b", "--direction", "-d",
        help="Translation direction for evaluation",
    ),
):
    """US 3.6 -- Multi-site generalisation test.

    Evaluate the trained CycleGAN on images from different manufacturing
    sites to assess cross-site generalisation.  Translates validation
    images from each site and computes quality metrics.
    """
    import tempfile

    import torch
    from rich.table import Table

    from udm_epic3.evaluation.quality_metrics import compute_ssim, evaluate_translation
    from udm_epic3.translation.translate import translate_directory

    cfg = _load_yaml(config)
    data_cfg = cfg.get("data", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect validation directories
    eval_sites: list[dict] = []

    # From config — use val splits
    for domain_key in ["domain_a", "domain_b"]:
        domain = data_cfg.get(domain_key, {})
        val_dir = domain.get("val")
        if val_dir and Path(val_dir).is_dir():
            eval_sites.append({"name": domain.get("name", domain_key), "dir": val_dir})

    # Additional site directories from CLI
    if site_dirs:
        for sd in site_dirs:
            p = Path(sd)
            if p.is_dir():
                eval_sites.append({"name": p.name, "dir": str(p)})
            else:
                rprint(f"[yellow]Warning: site directory not found: {sd}[/]")

    if not eval_sites:
        rprint("[bold red]No validation sites found. Check config or --site flags.[/]")
        raise typer.Exit(1)

    results_table = Table(title="Multi-Site Generalisation Results")
    results_table.add_column("Site", style="cyan")
    results_table.add_column("Images", justify="right")
    results_table.add_column("Mean SSIM", justify="right", style="green")
    results_table.add_column("Std SSIM", justify="right")

    for site in eval_sites:
        site_name = site["name"]
        site_dir = site["dir"]

        with tempfile.TemporaryDirectory() as tmpdir:
            translate_directory(
                model_path=checkpoint,
                input_dir=site_dir,
                output_dir=tmpdir,
                direction=direction,
                device=device,
            )

            df = evaluate_translation(
                real_dir=site_dir,
                translated_dir=tmpdir,
            )

            if df.empty:
                results_table.add_row(site_name, "0", "-", "-")
            else:
                results_table.add_row(
                    site_name,
                    str(len(df)),
                    f"{df['ssim'].mean():.4f}",
                    f"{df['ssim'].std():.4f}",
                )

    rprint(results_table)
    rprint("[bold green]Multi-site validation complete.[/]")


if __name__ == "__main__":
    app()
