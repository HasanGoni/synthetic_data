"""
UDM Epic 7 CLI — Chromasense multi-spectral: generate, preview, evaluate, visualize-spectrum.

Examples
--------
udm-epic7 generate         --config configs/epic7_chromasense.yaml
udm-epic7 preview          --config configs/epic7_chromasense.yaml --n-samples 5
udm-epic7 evaluate         --data-dir outputs/epic7_dataset --material copper
udm-epic7 visualize-spectrum --material copper --defect oxidation --severity 0.6
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic7",
    help="UDM Epic 7 — Chromasense Multi-Spectral Integration",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command("generate")
def generate(
    config: Path = typer.Option(
        "configs/epic7_chromasense.yaml", "--config", "-c",
        help="Chromasense config YAML",
    ),
    n_samples: int = typer.Option(1000, "--n-samples", "-n", help="Number of samples"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Generate a synthetic multi-spectral dataset."""
    from udm_epic7.spectral.wavelength_model import SpectralConfig
    from udm_epic7.data.dataset import generate_spectral_dataset

    cfg = _load_yaml(config)
    spec_cfg = cfg.get("spectral", {})
    render_cfg = cfg.get("rendering", {})

    wavelengths = spec_cfg.get("wavelengths", [450.0, 550.0, 650.0, 850.0])
    height = render_cfg.get("height", 512)
    width = render_cfg.get("width", 512)
    defect_prob = render_cfg.get("defect_prob", 0.7)
    seed = cfg.get("seed", 42)

    sc = SpectralConfig(wavelengths=wavelengths)
    out = output_dir or cfg.get("output", {}).get("dir", "outputs/epic7_dataset")

    out_path = generate_spectral_dataset(
        output_dir=out,
        n_samples=n_samples,
        config=sc,
        height=height,
        width=width,
        defect_prob=defect_prob,
        seed=seed,
    )
    rprint(f"[bold green]✓[/] Generated {n_samples} samples → [cyan]{out_path}[/]")


@app.command("preview")
def preview(
    config: Path = typer.Option(
        "configs/epic7_chromasense.yaml", "--config", "-c",
        help="Chromasense config YAML",
    ),
    n_samples: int = typer.Option(5, "--n-samples", "-n", help="Number of samples to preview"),
    output_dir: str = typer.Option("outputs/epic7_preview", "--output", "-o"),
):
    """Generate a few samples and save RGB previews."""
    import numpy as np

    from udm_epic7.spectral.wavelength_model import SpectralConfig
    from udm_epic7.data.dataset import SpectralDataset
    from udm_epic7.rendering.spectral_renderer import spectral_to_rgb

    cfg = _load_yaml(config)
    spec_cfg = cfg.get("spectral", {})
    render_cfg = cfg.get("rendering", {})
    wavelengths = spec_cfg.get("wavelengths", [450.0, 550.0, 650.0, 850.0])
    height = render_cfg.get("height", 512)
    width = render_cfg.get("width", 512)

    sc = SpectralConfig(wavelengths=wavelengths)
    ds = SpectralDataset(n_samples=n_samples, config=sc, height=height, width=width, seed=0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for i in range(n_samples):
        sample = ds[i]
        spectral = sample["image"].numpy()
        rgb = spectral_to_rgb(spectral, sc)
        Image.fromarray(rgb).save(out / f"preview_{i:03d}.png")
        rprint(
            f"  Sample {i}: defect=[bold]{sample['defect_type']}[/]  "
            f"mask_sum={sample['mask'].sum().item():.0f}"
        )

    rprint(f"\n[bold green]✓[/] Previews saved → [cyan]{out}[/]")


@app.command("evaluate")
def evaluate(
    data_dir: str = typer.Option(
        "outputs/epic7_dataset", "--data-dir", "-d",
        help="Directory with generated .npz samples",
    ),
    material: str = typer.Option("copper", "--material", "-m", help="Reference material"),
    n_samples: int = typer.Option(100, "--n-samples", "-n", help="Max samples to evaluate"),
):
    """Evaluate spectral anomaly detection on a generated dataset."""
    import numpy as np
    from rich.table import Table

    from udm_epic7.data.dataset import SpectralDataset
    from udm_epic7.evaluation.spectral_metrics import (
        spectral_anomaly_score_from_material,
    )

    ds = SpectralDataset(from_dir=data_dir)
    n = min(n_samples, len(ds))

    table = Table(title="Spectral Anomaly Evaluation")
    table.add_column("Sample", justify="right")
    table.add_column("Defect Type", style="cyan")
    table.add_column("Mean Anomaly", justify="right")
    table.add_column("Max Anomaly", justify="right")

    for i in range(n):
        sample = ds[i]
        image = sample["image"].numpy()
        scores = spectral_anomaly_score_from_material(image, material)
        table.add_row(
            str(i),
            sample["defect_type"],
            f"{scores.mean():.4f}",
            f"{scores.max():.4f}",
        )

    rprint(table)


@app.command("visualize-spectrum")
def visualize_spectrum(
    material: str = typer.Option("copper", "--material", "-m", help="Base material"),
    defect: Optional[str] = typer.Option(None, "--defect", help="Defect type to overlay"),
    severity: float = typer.Option(0.5, "--severity", "-s", help="Defect severity [0-1]"),
    output: str = typer.Option("outputs/epic7_spectrum.png", "--output", "-o"),
):
    """Plot the spectral reflectance curve for a material, with optional defect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from udm_epic7.spectral.wavelength_model import (
        SpectralConfig,
        default_spectral_config,
        material_reflectance,
    )
    from udm_epic7.spectral.defect_spectra import (
        delamination_spectrum,
        contamination_spectrum,
        oxidation_spectrum,
    )

    config = default_spectral_config()
    # Dense wavelength sampling for smooth plot
    wl_dense = np.linspace(
        min(config.wavelengths) - 20,
        max(config.wavelengths) + 20,
        200,
    )

    base_curve = [material_reflectance(material, float(w), config) for w in wl_dense]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl_dense, base_curve, "k-", linewidth=2, label=f"{material} (pristine)")

    if defect is not None:
        if defect == "delamination":
            defect_spec = delamination_spectrum(material, severity, config)
        elif defect == "contamination":
            defect_spec = contamination_spectrum("flux_residue", severity, material, config)
        elif defect == "oxidation":
            defect_spec = oxidation_spectrum(material, severity, config)
        else:
            rprint(f"[red]Unknown defect type: {defect}[/]")
            raise typer.Exit(1)

        dwl = sorted(defect_spec.keys())
        dref = [defect_spec[w] for w in dwl]
        ax.plot(dwl, dref, "ro--", linewidth=2, markersize=8, label=f"{defect} (severity={severity})")

    # Mark the Chromasense wavelengths
    for wl in config.wavelengths:
        ax.axvline(wl, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance", fontsize=12)
    ax.set_title("Chromasense Spectral Reflectance", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    rprint(f"[bold green]✓[/] Spectrum plot saved → [cyan]{out_path}[/]")


if __name__ == "__main__":
    app()
