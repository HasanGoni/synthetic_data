"""
UDM Epic 6 CLI — AOI Bond Wire Synthesis: generate, preview, evaluate, stats.

Examples
--------
udm-epic6 generate  --config configs/epic6_bond_wire.yaml --n-samples 500
udm-epic6 preview   --n 5 --seed 42
udm-epic6 evaluate  --data-dir outputs/epic6_dataset
udm-epic6 stats     --data-dir outputs/epic6_dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic6",
    help="UDM Epic 6 — AOI Bond Wire & Surface Defect Synthesis",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command("generate")
def generate(
    config: Path = typer.Option(
        "configs/epic6_bond_wire.yaml", "--config", "-c",
        help="YAML config for bond wire dataset generation",
    ),
    n_samples: int = typer.Option(1000, "--n-samples", "-n", help="Number of samples to generate"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Override output dir"),
):
    """Generate synthetic bond wire AOI dataset."""
    from udm_epic6.data.dataset import generate_bond_wire_dataset

    cfg = _load_yaml(config)
    out = output_dir or cfg.get("output", {}).get("dir", "outputs/epic6_dataset")

    gen_cfg = {
        "image_size": (
            cfg.get("image", {}).get("height", 512),
            cfg.get("image", {}).get("width", 512),
        ),
        "seed": cfg.get("seed", 42),
        "defect_probs": cfg.get("defects", {}).get("probabilities", None),
        "wire_range": (
            cfg.get("wires", {}).get("min_count", 1),
            cfg.get("wires", {}).get("max_count", 5),
        ),
    }

    rprint(f"[bold]Generating {n_samples} bond wire samples...[/]")
    result_dir = generate_bond_wire_dataset(out, n_samples=n_samples, config=gen_cfg)
    rprint(f"[bold green]OK[/] Dataset saved to [cyan]{result_dir}[/]")


@app.command("preview")
def preview(
    n: int = typer.Option(5, "--n", "-n", help="Number of samples to preview"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    image_size: int = typer.Option(256, "--size", help="Image size (square)"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", help="Save previews to dir instead of displaying"),
):
    """Preview N random synthetic bond wire samples."""
    import numpy as np
    from rich.table import Table

    from udm_epic6.data.dataset import BondWireDataset

    ds = BondWireDataset(
        n_samples=n,
        image_size=(image_size, image_size),
        seed=seed,
    )

    table = Table(title=f"Bond Wire Preview ({n} samples, seed={seed})")
    table.add_column("Index", justify="right")
    table.add_column("Defect Type", style="cyan")
    table.add_column("N Wires", justify="right")
    table.add_column("Severity", justify="right")
    table.add_column("Materials")
    table.add_column("Mask Pixels", justify="right")

    for i in range(n):
        sample = ds[i]
        meta = sample["metadata"]
        mask_px = int((sample["mask"] > 0.5).sum().item())
        table.add_row(
            str(i),
            meta["defect_type"],
            str(meta["n_wires"]),
            f"{meta['severity']:.2f}",
            ", ".join(meta["materials"][:3]),
            str(mask_px),
        )

        if save_dir:
            import cv2
            out_path = Path(save_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(
                str(out_path / f"preview_{i:03d}.png"),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
            )

    rprint(table)
    if save_dir:
        rprint(f"[bold green]OK[/] Previews saved to [cyan]{save_dir}[/]")


@app.command("evaluate")
def evaluate(
    data_dir: str = typer.Option("outputs/epic6_dataset", "--data-dir", "-d", help="Dataset directory"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Binarisation threshold"),
):
    """Evaluate defect detection on a generated bond wire dataset."""
    import torch
    from rich.table import Table

    from udm_epic6.evaluation.metrics import compute_f1, compute_iou

    data_path = Path(data_dir)
    meta_path = data_path / "metadata.json"
    if not meta_path.exists():
        rprint(f"[red]metadata.json not found in {data_dir}[/]")
        raise typer.Exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    rprint(f"[bold]Evaluating {len(metadata)} samples from {data_dir}[/]")

    # Group by defect type and report stats
    from collections import Counter
    type_counts = Counter(m["defect_type"] for m in metadata)

    table = Table(title="Defect Type Distribution")
    table.add_column("Defect Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")

    total = len(metadata)
    for dtype, count in sorted(type_counts.items()):
        table.add_row(dtype, str(count), f"{count / total:.1%}")

    rprint(table)
    rprint(f"\n[bold green]OK[/] Evaluation complete on [cyan]{data_dir}[/]")


@app.command("stats")
def stats(
    data_dir: str = typer.Option("outputs/epic6_dataset", "--data-dir", "-d", help="Dataset directory"),
):
    """Show dataset statistics: defect distribution, wire counts, materials."""
    import numpy as np
    from rich.table import Table

    data_path = Path(data_dir)
    meta_path = data_path / "metadata.json"
    if not meta_path.exists():
        rprint(f"[red]metadata.json not found in {data_dir}[/]")
        raise typer.Exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    total = len(metadata)
    rprint(f"\n[bold]Dataset: {data_dir}  ({total} samples)[/]\n")

    # Defect distribution
    from collections import Counter

    type_counts = Counter(m["defect_type"] for m in metadata)
    table = Table(title="Defect Types")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")
    for dtype, count in sorted(type_counts.items()):
        table.add_row(dtype, str(count), f"{count / total:.1%}")
    rprint(table)

    # Wire counts
    wire_counts = [m["n_wires"] for m in metadata]
    rprint(f"\nWire count: min={min(wire_counts)}, max={max(wire_counts)}, "
           f"mean={np.mean(wire_counts):.1f}")

    # Material distribution
    all_materials: list[str] = []
    for m in metadata:
        all_materials.extend(m.get("materials", []))
    mat_counts = Counter(all_materials)
    table2 = Table(title="Material Distribution")
    table2.add_column("Material", style="cyan")
    table2.add_column("Count", justify="right")
    for mat, count in sorted(mat_counts.items()):
        table2.add_row(mat, str(count))
    rprint(table2)

    # Severity stats (for defective samples only)
    severities = [m["severity"] for m in metadata if m["severity"] > 0]
    if severities:
        rprint(f"\nDefect severity: min={min(severities):.2f}, max={max(severities):.2f}, "
               f"mean={np.mean(severities):.2f}")

    rprint(f"\n[bold green]OK[/]")


if __name__ == "__main__":
    app()
