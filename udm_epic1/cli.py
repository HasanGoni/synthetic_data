"""
UDM Epic 1 — Command Line Interface

Usage:
    udm-generate run                     # use default config
    udm-generate run --config my.yaml   # use custom config
    udm-generate preview --n 4          # preview 4 samples, no save
    udm-generate validate               # validate output dataset
    udm-generate stats                  # print dataset statistics
"""

from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="udm-generate",
    help="UDM Epic 1 — Physics-based synthetic X-ray void dataset generator",
    add_completion=False,
)
console = Console()


@app.command("run")
def run(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to YAML config"),
    total: Optional[int] = typer.Option(None, "--total", "-n", help="Override total image count"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Override num_workers"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Override output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse config and show plan, no generation"),
):
    """
    Run the full synthetic dataset generation pipeline.

    Example:
        udm-generate run --config configs/default.yaml --total 3000 --workers 8
    """
    from udm_epic1.dataset.pipeline import DatasetPipeline, PipelineConfig

    rprint(f"[bold cyan]Loading config:[/] {config}")
    cfg = PipelineConfig.from_yaml(config)

    # Apply CLI overrides
    if total is not None:
        cfg.total_images = total
    if workers is not None:
        cfg.num_workers = workers
    if output is not None:
        cfg.output_dir = output

    # Show plan
    table = Table(title="Generation Plan", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total images", str(cfg.total_images))
    table.add_row("Train / Val / Test", f"{cfg.train_ratio:.0%} / {cfg.val_ratio:.0%} / {cfg.test_ratio:.0%}")
    table.add_row("Image size", f"{cfg.generator.height} × {cfg.generator.width}")
    table.add_row("Bit depth", str(cfg.bit_depth))
    table.add_row("Workers", str(cfg.num_workers))
    table.add_row("Output dir", cfg.output_dir)
    table.add_row("Seed", str(cfg.seed))
    console.print(table)

    if dry_run:
        rprint("[yellow]Dry run — no images generated.[/]")
        return

    pipeline = DatasetPipeline(cfg)
    manifest = pipeline.run()
    rprint(f"\n[bold green]✓ Done.[/] Manifest: [cyan]{manifest}[/]")


@app.command("preview")
def preview(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    n: int = typer.Option(4, "--n", help="Number of preview samples"),
    output: str = typer.Option("preview_samples", "--output", "-o"),
    seed: int = typer.Option(42, "--seed"),
):
    """
    Generate N preview samples and save as PNG for visual inspection.
    Does NOT write to the main dataset directory.

    Example:
        udm-generate preview --n 8 --output /tmp/preview
    """
    import numpy as np
    import cv2
    from pathlib import Path as P
    from udm_epic1.dataset.pipeline import PipelineConfig
    from udm_epic1.generators.sample_generator import SyntheticSampleGenerator

    cfg = PipelineConfig.from_yaml(config)
    out = P(output)
    out.mkdir(parents=True, exist_ok=True)

    gen = SyntheticSampleGenerator(cfg.generator, seed=seed)
    rprint(f"[cyan]Generating {n} preview samples → {out}[/]")

    for i in range(n):
        image, mask, meta = gen.generate(image_id=f"preview_{i:04d}", split="preview")

        # Convert 16-bit → 8-bit for easy viewing
        img8 = (image.astype(float) / 65535.0 * 255).astype(np.uint8) if image.dtype == np.uint16 else image

        # Save image
        cv2.imwrite(str(out / f"preview_{i:04d}_image.png"), img8)
        # Save mask
        cv2.imwrite(str(out / f"preview_{i:04d}_mask.png"), mask)

        # Overlay: green void contours on image
        overlay = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(out / f"preview_{i:04d}_overlay.png"), overlay)

        rprint(f"  [{i+1}/{n}] {meta.image_id}: {meta.n_voids} voids, area={meta.total_void_area_fraction:.4f}")

    rprint(f"[bold green]✓ Saved {n} previews to {out}[/]")


@app.command("validate")
def validate(
    output: str = typer.Option("data/synthetic", "--output", "-o", help="Dataset directory to validate"),
):
    """
    Validate the generated dataset:
      - Check all image/mask pairs exist
      - Check image/mask size consistency
      - Check mask values are binary (0 or 255)
      - Report split counts
    """
    import numpy as np
    import cv2
    from pathlib import Path as P

    base = P(output)
    splits = ["train", "val", "test"]
    total_ok = 0
    total_err = 0

    for split in splits:
        img_dir = base / split / "images"
        msk_dir = base / split / "masks"
        if not img_dir.exists():
            rprint(f"[yellow]Split '{split}' not found, skipping.[/]")
            continue

        imgs = sorted(img_dir.glob("*.png"))
        ok = 0
        err = 0
        for img_path in imgs:
            msk_path = msk_dir / img_path.name
            if not msk_path.exists():
                rprint(f"[red]Missing mask: {msk_path}[/]")
                err += 1
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
            if img is None or msk is None:
                rprint(f"[red]Cannot read: {img_path}[/]")
                err += 1
                continue
            if img.shape != msk.shape:
                rprint(f"[red]Shape mismatch: {img_path}[/]")
                err += 1
                continue
            unique = np.unique(msk)
            if not all(v in (0, 255) for v in unique):
                rprint(f"[red]Non-binary mask values: {img_path} → {unique}[/]")
                err += 1
                continue
            ok += 1

        total_ok += ok
        total_err += err
        status = "[green]✓[/]" if err == 0 else "[red]✗[/]"
        rprint(f"  {status} {split}: {ok} OK, {err} errors")

    rprint(f"\n[bold]Total: {total_ok} OK, {total_err} errors[/]")
    if total_err > 0:
        raise typer.Exit(code=1)


@app.command("stats")
def stats(
    output: str = typer.Option("data/synthetic", "--output", "-o"),
):
    """Print statistics from the manifest CSV."""
    import pandas as pd
    from pathlib import Path as P

    manifest = P(output) / "manifest.csv"
    if not manifest.exists():
        rprint(f"[red]Manifest not found: {manifest}[/]")
        raise typer.Exit(code=1)

    df = pd.read_csv(manifest)
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total samples", str(len(df)))
    for split in ["train", "val", "test"]:
        n = len(df[df["split"] == split])
        table.add_row(f"  {split}", str(n))
    table.add_row("With voids", f"{df['has_voids'].sum()} ({100*df['has_voids'].mean():.1f}%)")
    table.add_row("Mean voids/image", f"{df['n_voids'].mean():.2f}")
    table.add_row("Max voids/image", str(df["n_voids"].max()))
    table.add_row("Mean void area", f"{100*df['total_void_area_fraction'].mean():.4f}%")
    table.add_row("Max void area", f"{100*df['total_void_area_fraction'].max():.4f}%")

    console.print(table)


if __name__ == "__main__":
    app()
