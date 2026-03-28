"""
UDM Epic 2 CLI — dataset prep, HF export, training, generation, paste.

Examples
--------
udm-epic2 extract-crops --image-dir ... --mask-dir ... -o data/epic2/crops -c configs/epic2_dataset.yaml
udm-epic2 train --config configs/epic2_train.yaml
udm-epic2 generate --config configs/epic2_generate.yaml
udm-epic2 paste --background bg.png --patch patch.png --mask mask.png --center-x 256 --center-y 256 -o out.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic2",
    help="UDM Epic 2 — ControlNet data prep, training, generation, paste",
    no_args_is_help=True,
)


def _load_yaml_config(path: Optional[Path]) -> dict:
    if path is None or not path.is_file():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command("extract-crops")
def extract_crops(
    image_dir: Path = typer.Option(..., "--image-dir", help="Folder of source images (PNG)"),
    mask_dir: Path = typer.Option(..., "--mask-dir", help="Folder of void masks (same filenames)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output root for crops + manifest.csv"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML (configs/epic2_dataset.yaml)"),
    defect_class: Optional[str] = typer.Option(None, "--defect-class"),
    glob: Optional[str] = typer.Option(None, "--glob"),
    min_area: Optional[int] = typer.Option(None, "--min-area"),
    padding: Optional[int] = typer.Option(None, "--padding"),
    max_side: Optional[int] = typer.Option(None, "--max-side"),
    write_edges: Optional[bool] = typer.Option(None, "--write-edges/--no-edges"),
):
    """US 2.1 / 2.2 — defect crops + optional Canny edge maps."""
    from udm_epic2.dataset.crops import CropConfig, process_crop_dataset

    raw = _load_yaml_config(config)
    crop_raw = raw.get("crop", {}) if isinstance(raw, dict) else {}
    io_raw = raw.get("io", {}) if isinstance(raw, dict) else {}

    cfg = CropConfig(
        min_component_area_px=int(crop_raw.get("min_component_area_px", 16)),
        padding_px=int(crop_raw.get("padding_px", 8)),
        max_crop_side=int(crop_raw.get("max_crop_side", 512)),
        defect_class=str(crop_raw.get("defect_class", "void")),
    )
    if defect_class is not None:
        cfg.defect_class = defect_class
    if min_area is not None:
        cfg.min_component_area_px = min_area
    if padding is not None:
        cfg.padding_px = padding
    if max_side is not None:
        cfg.max_crop_side = max_side

    g = glob if glob is not None else str(io_raw.get("glob", "*.png"))
    edges = write_edges if write_edges is not None else bool(io_raw.get("write_edges", False))

    manifest = process_crop_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        out_root=output,
        cfg=cfg,
        glob=g,
        write_edges=edges,
    )
    rprint(f"[bold green]✓[/] Wrote manifest: [cyan]{manifest}[/]")


@app.command("export-hf")
def export_hf(
    crops_root: Path = typer.Option(..., "--crops-root", help="Epic 2 crop root (with manifest + edges)"),
    output: Path = typer.Option(..., "--output", "-o", help="HF-style folder output"),
    caption: str = typer.Option(
        "semiconductor x-ray void defect, high contrast",
        "--caption",
        help="Text prompt column for each row",
    ),
):
    """Optional — export ``train/image`` + ``train/conditioning_image`` + metadata for HF scripts."""
    from udm_epic2.dataset.hf_export import export_hf_style_folder

    meta = export_hf_style_folder(crops_root, output, caption=caption)
    rprint(f"[bold green]✓[/] HF metadata: [cyan]{meta}[/]")


@app.command("train")
def train_cmd(
    config: Path = typer.Option(
        "configs/epic2_train.yaml",
        "--config",
        "-c",
        help="Training YAML (requires pip install -e \".[epic2]\")",
    ),
):
    """US 2.3 — fine-tune ControlNet on Epic 2 crops + edges."""
    from udm_epic2.training.controlnet_train import train_controlnet_from_yaml

    train_controlnet_from_yaml(config)


@app.command("generate")
def generate_cmd(
    config: Path = typer.Option("configs/epic2_generate.yaml", "--config", "-c"),
):
    """US 2.4 — generate defect images from edge maps using fine-tuned ControlNet."""
    import glob as glob_mod

    with open(config) as f:
        cfg = yaml.safe_load(f) or {}

    from udm_epic2.generation.inference import generate_defect_samples

    paths = sorted(glob_mod.glob(str(cfg["conditioning_glob"])))
    if not paths:
        raise typer.BadParameter(f"No files match conditioning_glob: {cfg['conditioning_glob']}")

    out = generate_defect_samples(
        controlnet_path=Path(cfg["controlnet_path"]),
        conditioning_image_paths=[Path(p) for p in paths],
        output_dir=Path(cfg["output_dir"]),
        pretrained_model_name_or_path=str(cfg.get("pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5")),
        prompt=str(cfg.get("prompt", "semiconductor x-ray void defect")),
        num_inference_steps=int(cfg.get("num_inference_steps", 30)),
        n_per_conditioning=int(cfg.get("n_per_conditioning", 1)),
        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
        quality_filter=bool(cfg.get("quality_filter", True)),
        min_laplacian_var=float(cfg.get("min_laplacian_var", 5.0)),
    )
    rprint(f"[bold green]✓[/] Saved {len(out)} images → [cyan]{cfg['output_dir']}[/]")


@app.command("paste")
def paste_cmd(
    background: Path = typer.Option(..., "--background", "-b", help="Full background image (OK sample)"),
    patch: Path = typer.Option(..., "--patch", "-p", help="Defect patch (same size as mask)"),
    mask: Path = typer.Option(..., "--mask", "-m", help="Defect mask for patch"),
    center_x: int = typer.Option(..., "--center-x", help="Paste center x in background"),
    center_y: int = typer.Option(..., "--center-y", help="Paste center y in background"),
    output: Path = typer.Option(..., "--output", "-o", help="Output image path"),
    mode: str = typer.Option("poisson", "--mode", help="poisson or alpha"),
):
    """US 2.5 — paste synthetic defect onto real background (Poisson / alpha)."""
    import cv2

    from udm_epic2.integration.paste import paste_defect_on_background

    bg = cv2.imread(str(background), cv2.IMREAD_UNCHANGED)
    pt = cv2.imread(str(patch), cv2.IMREAD_UNCHANGED)
    mk = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if bg is None or pt is None or mk is None:
        raise typer.BadParameter("Could not read one of the input images")
    out = paste_defect_on_background(bg, pt, mk, (center_x, center_y), mode=mode)  # type: ignore[arg-type]
    cv2.imwrite(str(output), out)
    rprint(f"[bold green]✓[/] Wrote [cyan]{output}[/]")


@app.command("ablation-template")
def ablation_template(
    output: Path = typer.Option(
        "results/epic2_ablation_template.csv",
        "--output",
        "-o",
        help="CSV template for US 2.6 synthetic-ratio experiments",
    ),
):
    """US 2.6 — write a CSV template for F1 vs synthetic ratio (0–100%)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "synthetic_ratio_pct,model_name,notes,f1,precision,recall\n"
        "0,baseline_real_only,train only on real,\n"
        "25,,25pct synthetic in train mix,\n"
        "50,,,\n"
        "75,,,\n"
        "100,synthetic_only,,\n"
    )
    output.write_text(text)
    rprint(f"[bold green]✓[/] Template: [cyan]{output}[/]")


if __name__ == "__main__":
    app()
