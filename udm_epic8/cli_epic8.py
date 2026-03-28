"""
UDM Epic 8 CLI -- Universal model support: generate, list, export, merge, report.

Examples
--------
udm-epic8 generate       --config configs/epic8_universal.yaml
udm-epic8 generate       --modality xray --samples 100 --output data/xray_out
udm-epic8 list-modalities
udm-epic8 export         --data-dir data/universal/xray --format coco --output exports/coco.json
udm-epic8 merge          --dirs data/universal/xray data/universal/aoi --output data/merged
udm-epic8 report         --results results.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich import print as rprint
from rich.table import Table

app = typer.Typer(
    name="udm-epic8",
    help="UDM Epic 8 -- Universal Model Support & Integration",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── generate ──────────────────────────────────────────────────────────────────


@app.command("generate")
def generate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Universal config YAML (configs/epic8_universal.yaml)",
    ),
    modality: Optional[str] = typer.Option(
        None, "--modality", "-m",
        help="Single modality to generate (overrides config)",
    ),
    samples: int = typer.Option(
        100, "--samples", "-n",
        help="Number of samples (used with --modality)",
    ),
    output: str = typer.Option(
        "data/universal", "--output", "-o",
        help="Output directory",
    ),
):
    """Generate synthetic data for one or all modalities."""
    from udm_epic8.pipeline.unified import UnifiedPipeline, UnifiedPipelineConfig

    if config is not None:
        cfg_dict = _load_yaml(config)
        modalities_cfg = cfg_dict.get("modalities", {})
        enabled = [
            name for name, mcfg in modalities_cfg.items()
            if mcfg.get("enabled", True)
        ]
        per_mod = {}
        for name, mcfg in modalities_cfg.items():
            per_mod[name] = {
                "samples": mcfg.get("samples", 100),
                "config_ref": mcfg.get("config_ref"),
            }
        out_cfg = cfg_dict.get("output", {})
        pipeline_config = UnifiedPipelineConfig(
            modalities=enabled,
            per_modality_config=per_mod,
            output_dir=out_cfg.get("dir", output),
            total_samples=samples,
            train_ratio=out_cfg.get("train_ratio", 0.75),
            val_ratio=out_cfg.get("val_ratio", 0.15),
            test_ratio=out_cfg.get("test_ratio", 0.10),
        )
    elif modality is not None:
        pipeline_config = UnifiedPipelineConfig(
            modalities=[modality],
            per_modality_config={modality: {"samples": samples}},
            output_dir=output,
            total_samples=samples,
        )
    else:
        rprint("[red]Provide --config or --modality[/]")
        raise typer.Exit(1)

    pipeline = UnifiedPipeline(pipeline_config)
    manifest = pipeline.run()
    rprint(f"\n[bold green]OK[/] Manifest written -> [cyan]{manifest}[/]")


# ── list-modalities ───────────────────────────────────────────────────────────


@app.command("list-modalities")
def list_modalities():
    """List all available modality generators."""
    from udm_epic8.registry.modality_registry import registry

    table = Table(title="Registered Modalities")
    table.add_column("Name", style="cyan")
    table.add_column("Config Class", style="green")

    for name in registry.list_modalities():
        _fn, cfg_cls = registry.get(name)
        table.add_row(name, cfg_cls.__name__)

    rprint(table)


# ── export ────────────────────────────────────────────────────────────────────


@app.command("export")
def export(
    data_dir: str = typer.Option(..., "--data-dir", "-d", help="Source dataset directory"),
    fmt: str = typer.Option("coco", "--format", "-f", help="Export format: coco, yolo, hf"),
    output: str = typer.Option(..., "--output", "-o", help="Output path (file or directory)"),
    modality: Optional[str] = typer.Option(None, "--modality", help="Modality label for COCO"),
):
    """Export dataset to COCO, YOLO or HuggingFace format."""
    from udm_epic8.export.dataset_export import export_to_coco, export_to_hf, export_to_yolo

    fmt_lower = fmt.lower()
    if fmt_lower == "coco":
        result = export_to_coco(data_dir, output, modality=modality)
    elif fmt_lower == "yolo":
        result = export_to_yolo(data_dir, output)
    elif fmt_lower in ("hf", "huggingface"):
        result = export_to_hf(data_dir, output)
    else:
        rprint(f"[red]Unknown format '{fmt}'. Choose: coco, yolo, hf[/]")
        raise typer.Exit(1)

    rprint(f"[bold green]OK[/] Exported to {fmt_lower} -> [cyan]{result}[/]")


# ── merge ─────────────────────────────────────────────────────────────────────


@app.command("merge")
def merge(
    dirs: List[str] = typer.Option(..., "--dirs", "-d", help="Directories to merge"),
    output: str = typer.Option("data/merged", "--output", "-o", help="Merged output directory"),
):
    """Merge datasets from multiple modalities into one."""
    from udm_epic8.export.dataset_export import merge_datasets

    result = merge_datasets(dirs, output)
    rprint(f"[bold green]OK[/] Merged manifest -> [cyan]{result}[/]")


# ── report ────────────────────────────────────────────────────────────────────


@app.command("report")
def report(
    results: Optional[Path] = typer.Option(
        None, "--results", "-r",
        help="JSON file with per-modality results dict",
    ),
    data_dir: Optional[str] = typer.Option(
        None, "--data-dir", "-d",
        help="Universal pipeline output dir to scan for manifests",
    ),
):
    """Generate a cross-modality evaluation report."""
    import json

    from udm_epic8.evaluation.cross_modality import cross_modality_report

    if results is not None:
        with open(results) as f:
            results_dict = json.load(f)
    elif data_dir is not None:
        # Scan manifests in sub-directories
        data_path = Path(data_dir)
        results_dict = {}
        for manifest_file in sorted(data_path.rglob("manifest.json")):
            if manifest_file.parent == data_path:
                continue  # skip top-level manifest
            with open(manifest_file) as f:
                mod_manifest = json.load(f)
            mod_name = mod_manifest.get("modality", manifest_file.parent.name)
            results_dict[mod_name] = {
                "n_samples": mod_manifest.get("n_samples", 0),
            }
    else:
        rprint("[red]Provide --results or --data-dir[/]")
        raise typer.Exit(1)

    df = cross_modality_report(results_dict)
    rprint(df.to_string(index=False))
    rprint(f"\n[bold green]OK[/] Report generated for {len(df)} modalities")


if __name__ == "__main__":
    app()
