"""
UDM Epic 4 CLI — DANN domain adaptation: prepare, baseline, train, analyze, evaluate, report.

Examples
--------
udm-epic4 prepare   --config configs/epic4_data.yaml
udm-epic4 baseline  --config configs/epic4_baseline.yaml
udm-epic4 train     --config configs/epic4_dann.yaml
udm-epic4 analyze   --checkpoint outputs/epic4_dann/best.pth --config configs/epic4_dann.yaml
udm-epic4 evaluate  --checkpoint outputs/epic4_dann/best.pth --config configs/epic4_evaluate.yaml
udm-epic4 report    --checkpoint outputs/epic4_dann/best.pth --config configs/epic4_evaluate.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic4",
    help="UDM Epic 4 — Domain Adversarial Neural Networks for multi-site deployment",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@app.command("prepare")
def prepare(
    config: Path = typer.Option(
        "configs/epic4_data.yaml", "--config", "-c",
        help="Data config YAML with domain paths",
    ),
):
    """US 4.1 — Analyze domain shift: dataset stats, intensity histograms, feature t-SNE."""
    import numpy as np
    import torch
    from rich.table import Table

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic4_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    from udm_epic4.data.multi_domain_dataset import DomainDataset

    # Collect stats per domain
    table = Table(title="Domain Dataset Summary")
    table.add_column("Domain", style="cyan")
    table.add_column("Images", justify="right")
    table.add_column("Has Masks", justify="center")
    table.add_column("Mean Intensity", justify="right")
    table.add_column("Std Intensity", justify="right")

    all_domains = []
    src = cfg["domains"]["source"]
    all_domains.append({"name": src["name"], "images": src["images"], "masks": src.get("masks")})
    for t in cfg["domains"].get("targets", []):
        all_domains.append({"name": t["name"], "images": t["images"], "masks": t.get("masks")})
    for e in cfg["domains"].get("evaluation", []):
        all_domains.append({"name": e["name"], "images": e["images"], "masks": e.get("masks")})

    for dom in all_domains:
        img_dir = Path(dom["images"])
        mask_dir = Path(dom["masks"]) if dom.get("masks") else None
        if not img_dir.is_dir():
            table.add_row(dom["name"], "[red]NOT FOUND[/]", "-", "-", "-")
            continue
        ds = DomainDataset(
            images_dir=str(img_dir),
            masks_dir=str(mask_dir) if mask_dir else None,
            domain_label=0,
            image_size=(h, w),
        )
        n = len(ds)
        if n == 0:
            table.add_row(dom["name"], "0", str(mask_dir is not None), "-", "-")
            continue
        # Sample up to 50 images for stats
        sample_n = min(50, n)
        intensities = []
        for i in range(sample_n):
            sample = ds[i]
            intensities.append(sample["image"].mean().item())
        mean_i = np.mean(intensities)
        std_i = np.std(intensities)
        table.add_row(
            dom["name"], str(n), str(mask_dir is not None),
            f"{mean_i:.4f}", f"{std_i:.4f}",
        )

    rprint(table)
    rprint(f"\n[bold green]✓[/] Analysis output → [cyan]{out_dir}[/]")


@app.command("baseline")
def baseline(
    config: Path = typer.Option(
        "configs/epic4_baseline.yaml", "--config", "-c",
        help="Baseline training config",
    ),
):
    """US 4.2 — Train source-only segmentation model (no domain adaptation)."""
    from udm_epic4.training.train_baseline import train_baseline_from_yaml

    best_ckpt = train_baseline_from_yaml(config)
    rprint(f"[bold green]✓[/] Best checkpoint: [cyan]{best_ckpt}[/]")


@app.command("train")
def train(
    config: Path = typer.Option(
        "configs/epic4_dann.yaml", "--config", "-c",
        help="DANN training config",
    ),
):
    """US 4.3 — Train DANN with gradient reversal layer."""
    from udm_epic4.training.train_dann import train_dann_from_yaml

    best_ckpt = train_dann_from_yaml(config)
    rprint(f"[bold green]✓[/] Best DANN checkpoint: [cyan]{best_ckpt}[/]")


@app.command("analyze")
def analyze(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint (.pth)"),
    config: Path = typer.Option(
        "configs/epic4_dann.yaml", "--config", "-c",
        help="Config with data paths",
    ),
    max_samples: int = typer.Option(500, "--max-samples", help="Max samples for t-SNE"),
):
    """US 4.4 — Domain confusion analysis: t-SNE, domain classifier accuracy."""
    import torch

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic4_dann")) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    from udm_epic4.models.dann import DANNModel
    from udm_epic4.data.multi_domain_dataset import DomainDataset
    from udm_epic4.evaluation.domain_analysis import (
        extract_features, compute_tsne, domain_confusion_score, plot_tsne,
    )

    model_cfg = cfg.get("model", {})
    model = DANNModel(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        pretrained=False,
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32]),
        domain_head_hidden=model_cfg.get("domain_head_hidden", 256),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()

    # Build source + target datasets
    src_cfg = cfg["data"]["source"]
    source_ds = DomainDataset(src_cfg["images"], src_cfg.get("masks"), 0, image_size=(h, w))

    target_cfgs = cfg["data"].get("targets", [])
    target_ds = None
    if target_cfgs:
        t = target_cfgs[0]
        target_ds = DomainDataset(t["images"], t.get("masks"), 1, image_size=(h, w))

    # Extract features & t-SNE
    import numpy as np

    feats_s, labels_s = extract_features(model, source_ds, device=device, max_samples=max_samples)
    domain_names = [src_cfg["name"]]
    all_feats = [feats_s]
    all_labels = [labels_s]

    if target_ds and len(target_ds) > 0:
        feats_t, labels_t = extract_features(model, target_ds, device=device, max_samples=max_samples)
        all_feats.append(feats_t)
        all_labels.append(labels_t)
        domain_names.append(target_cfgs[0]["name"])

    combined_feats = np.concatenate(all_feats)
    combined_labels = np.concatenate(all_labels)

    coords = compute_tsne(combined_feats)
    tsne_path = str(out_dir / "tsne_domains.png")
    plot_tsne(coords, combined_labels, domain_names, save_path=tsne_path)
    rprint(f"[bold green]✓[/] t-SNE saved: [cyan]{tsne_path}[/]")

    # Domain confusion
    if target_ds and len(target_ds) > 0:
        acc = domain_confusion_score(model, source_ds, target_ds, device=device)
        rprint(f"Domain classifier accuracy: [bold]{acc:.1%}[/] (lower = better DANN)")
        if acc < 0.6:
            rprint("[green]✓ Domain confusion achieved (<60%)[/]")
        else:
            rprint("[yellow]⚠ Domain classifier still accurate — DANN may need more training[/]")


@app.command("evaluate")
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint (.pth)"),
    config: Path = typer.Option(
        "configs/epic4_evaluate.yaml", "--config", "-c",
        help="Evaluation config with domain paths",
    ),
):
    """US 4.5 — Evaluate on all domains, compute F1/IoU/Dice per site."""
    import torch

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic4_evaluation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    from udm_epic4.models.dann import DANNModel
    from udm_epic4.data.multi_domain_dataset import DomainDataset
    from udm_epic4.evaluation.metrics import evaluate_all_domains

    model_cfg = cfg.get("model", {})
    model = DANNModel(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        pretrained=False,
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32]),
        domain_head_hidden=model_cfg.get("domain_head_hidden", 256),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()

    eval_datasets = []
    for i, ev in enumerate(cfg.get("evaluation", [])):
        ds = DomainDataset(ev["images"], ev.get("masks"), i, image_size=(h, w))
        ds.domain_name = ev["name"]
        eval_datasets.append(ds)

    df = evaluate_all_domains(model, eval_datasets, device=device)

    csv_path = out_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    rprint(df.to_string(index=False))
    rprint(f"\n[bold green]✓[/] Results saved: [cyan]{csv_path}[/]")


@app.command("report")
def report(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Model checkpoint (.pth)"),
    config: Path = typer.Option(
        "configs/epic4_evaluate.yaml", "--config", "-c",
        help="Evaluation config",
    ),
    n_visual: int = typer.Option(20, "--n-visual", help="Number of failure examples to save"),
):
    """US 4.6 — Generate failure analysis: categorize errors, visual report."""
    import torch

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic4_evaluation")) / "failure_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    from udm_epic4.models.dann import DANNModel
    from udm_epic4.data.multi_domain_dataset import DomainDataset
    from udm_epic4.reporting.failure_analysis import categorize_failures, generate_failure_report

    model_cfg = cfg.get("model", {})
    model = DANNModel(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        pretrained=False,
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32]),
        domain_head_hidden=model_cfg.get("domain_head_hidden", 256),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()

    for ev in cfg.get("evaluation", []):
        ds = DomainDataset(ev["images"], ev.get("masks"), 0, image_size=(h, w))
        if len(ds) == 0:
            continue
        domain_out = out_dir / ev["name"]
        domain_out.mkdir(parents=True, exist_ok=True)

        failures_df = categorize_failures(model, ds, device=device)
        generate_failure_report(
            failures_df, str(domain_out),
            model=model, dataset=ds, device=device, n_visual=n_visual,
        )
        rprint(f"[bold green]✓[/] {ev['name']}: report → [cyan]{domain_out}[/]")

    rprint(f"\n[bold green]✓[/] All failure reports → [cyan]{out_dir}[/]")


if __name__ == "__main__":
    app()
