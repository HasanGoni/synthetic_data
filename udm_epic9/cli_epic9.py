"""
UDM Epic 9 CLI — Synthetic Crack Generation: generate, from-mask, transfer,
preview, evaluate, stats.

Examples
--------
udm-epic9 generate   --config configs/epic9_crack.yaml
udm-epic9 from-mask  --mask crack_mask.png --domain rgb --output out.png
udm-epic9 transfer   --input-dir data/usm/ --output-dir data/rgb/
udm-epic9 preview    --n 8 --domain both --output preview.png
udm-epic9 evaluate   --checkpoint model.pth --config configs/epic9_crack.yaml
udm-epic9 stats      --config configs/epic9_crack.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic9",
    help="UDM Epic 9 — Synthetic Crack Generation for semiconductor defect detection",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command("generate")
def generate(
    config: Path = typer.Option(
        "configs/epic9_crack.yaml", "--config", "-c",
        help="Crack generation config YAML",
    ),
    n_samples: int = typer.Option(0, "--n-samples", "-n", help="Override n_samples (0=use config)"),
    domain: str = typer.Option("both", "--domain", "-d", help="Domain: usm, rgb, or both"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Override output directory"),
):
    """Generate synthetic crack dataset (USM and/or RGB)."""
    from udm_epic9.data.crack_dataset import generate_crack_dataset

    cfg = _load_yaml(config)

    if n_samples > 0:
        cfg.setdefault("dataset", {})["n_samples"] = n_samples

    domains = {"usm": ["usm"], "rgb": ["rgb"], "both": ["usm", "rgb"]}[domain]
    cfg.setdefault("dataset", {})["domains"] = domains

    out_dir = output or cfg.get("output", {}).get("dir", "data/epic9_cracks")

    manifest = generate_crack_dataset(output_dir=out_dir, config=cfg)
    rprint(f"[bold green]OK[/] Generated dataset -> [cyan]{manifest}[/]")


@app.command("from-mask")
def from_mask(
    mask: Path = typer.Option(..., "--mask", "-m", help="Input crack mask image (PNG)"),
    domain: str = typer.Option("usm", "--domain", "-d", help="Target domain: usm or rgb"),
    output: Path = typer.Option("crack_output.png", "--output", "-o", help="Output image path"),
):
    """Generate image from crack mask (mask-to-image capability)."""
    import cv2
    import numpy as np

    from udm_epic9.domain_transfer.usm_to_rgb import mask_to_image

    crack_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if crack_mask is None:
        rprint(f"[bold red]Error[/] Cannot read mask: {mask}")
        raise typer.Exit(1)

    result = mask_to_image(crack_mask, target_domain=domain)

    if domain == "usm":
        save_img = np.clip(result * 255, 0, 255).astype(np.uint8)
    else:
        save_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output), save_img)
    rprint(f"[bold green]OK[/] {domain.upper()} image -> [cyan]{output}[/]")


@app.command("transfer")
def transfer(
    input_dir: Path = typer.Option(..., "--input-dir", "-i", help="Directory of USM images"),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Output directory for RGB images"),
    method: str = typer.Option("colormap", "--method", "-m", help="Transfer method: colormap or learned"),
):
    """Domain transfer USM -> RGB for a directory of images."""
    import cv2
    import numpy as np

    from udm_epic9.domain_transfer.usm_to_rgb import USMtoRGBTransfer

    output_dir.mkdir(parents=True, exist_ok=True)
    transferer = USMtoRGBTransfer(method=method)

    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        rprint(f"[yellow]No PNG files found in {input_dir}[/]")
        raise typer.Exit(1)

    for img_path in image_files:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        usm_f = gray.astype(np.float32) / 255.0
        rgb = transferer.transfer(usm_f)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / img_path.name), bgr)

    rprint(f"[bold green]OK[/] Transferred {len(image_files)} images -> [cyan]{output_dir}[/]")


@app.command("preview")
def preview(
    n: int = typer.Option(4, "--n", help="Number of random samples to preview"),
    domain: str = typer.Option("both", "--domain", "-d", help="Domain: usm, rgb, or both"),
    output: Path = typer.Option("epic9_preview.png", "--output", "-o", help="Output preview image"),
):
    """Show N random crack samples as a grid image."""
    import numpy as np

    from udm_epic9.domain_transfer.usm_to_rgb import USMtoRGBTransfer
    from udm_epic9.rendering.usm_renderer import generate_synthetic_usm_with_cracks

    rng = np.random.default_rng()
    transferer = USMtoRGBTransfer(method="colormap")

    rows = []
    for i in range(n):
        usm_img, mask, meta = generate_synthetic_usm_with_cracks(
            height=256, width=256, rng=rng,
        )

        # Convert USM to displayable uint8 RGB
        usm_disp = np.stack([np.clip(usm_img * 255, 0, 255).astype(np.uint8)] * 3, axis=-1)
        mask_disp = np.stack([mask] * 3, axis=-1)

        panels = []
        if domain in ("usm", "both"):
            panels.extend([usm_disp, mask_disp])
        if domain in ("rgb", "both"):
            rgb_img = transferer.transfer_with_cracks(usm_img, mask)
            panels.append(rgb_img)

        row = np.concatenate(panels, axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    import cv2
    cv2.imwrite(str(output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    rprint(f"[bold green]OK[/] Preview ({n} samples) -> [cyan]{output}[/]")


@app.command("evaluate")
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint (.pth)"),
    config: Path = typer.Option(
        "configs/epic9_crack.yaml", "--config", "-c",
        help="Config YAML",
    ),
):
    """Evaluate model on crack test set."""
    import torch

    from udm_epic9.data.crack_dataset import CrackDataset
    from udm_epic9.evaluation.crack_metrics import evaluate_crack_dataset

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "data/epic9_cracks"))
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    test_images = out_dir / "test" / "usm" / "images"
    test_masks = out_dir / "test" / "usm" / "masks"

    dataset = CrackDataset(
        images_dir=str(test_images) if test_images.is_dir() else None,
        masks_dir=str(test_masks) if test_masks.is_dir() else None,
        image_size=(h, w),
        domain="usm",
        n_samples=50,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if isinstance(model, dict) and "model_state_dict" in model:
        rprint("[yellow]Checkpoint is a state_dict — load your model architecture first.[/]")
        raise typer.Exit(1)

    df = evaluate_crack_dataset(model, dataset, device=device)
    rprint(df.describe().to_string())

    csv_path = out_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    rprint(f"\n[bold green]OK[/] Results -> [cyan]{csv_path}[/]")


@app.command("stats")
def stats(
    config: Path = typer.Option(
        "configs/epic9_crack.yaml", "--config", "-c",
        help="Config YAML",
    ),
):
    """Dataset statistics — sample counts, crack type distribution, splits."""
    import json

    from rich.table import Table

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "data/epic9_cracks"))
    manifest_path = out_dir / "manifest.json"

    if not manifest_path.exists():
        rprint(f"[yellow]No manifest found at {manifest_path}. Run 'generate' first.[/]")
        raise typer.Exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    table = Table(title="Epic 9 Crack Dataset Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total samples", str(manifest["n_samples"]))
    table.add_row("Domains", ", ".join(manifest["domains"]))
    table.add_row("Image size", f"{manifest['image_size'][0]}x{manifest['image_size'][1]}")
    table.add_row("Seed", str(manifest["seed"]))

    for split, count in manifest["splits"].items():
        table.add_row(f"  {split}", str(count))

    # Crack type distribution
    type_counts: dict[str, int] = {}
    for sample in manifest["samples"]:
        ct = sample["crack_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    table.add_row("---", "---")
    for ct, count in sorted(type_counts.items()):
        table.add_row(f"  {ct}", str(count))

    rprint(table)


if __name__ == "__main__":
    app()
