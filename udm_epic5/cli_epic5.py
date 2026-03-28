"""
UDM Epic 5 CLI — Active Domain Adaptation: uncertainty, select, label, train, analyze.

Examples
--------
udm-epic5 uncertainty   --checkpoint outputs/epic4_dann/best.pth --config configs/epic5_active.yaml
udm-epic5 select        --uncertainty-csv outputs/epic5_active/uncertainty.csv --config configs/epic5_active.yaml
udm-epic5 prepare-session --selected-csv outputs/epic5_active/selected.csv --images-dir data/malaysia/images
udm-epic5 train         --config configs/epic5_active.yaml
udm-epic5 analyze       --results-csv outputs/epic5_active/results.csv
udm-epic5 run-round     --checkpoint outputs/epic4_dann/best.pth --config configs/epic5_active.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

app = typer.Typer(
    name="udm-epic5",
    help="UDM Epic 5 — Active Domain Adaptation",
    no_args_is_help=True,
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ======================================================================
# US 5.1 — Uncertainty estimation via MC Dropout
# ======================================================================


@app.command("uncertainty")
def uncertainty(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Model checkpoint (.pth)",
    ),
    config: Path = typer.Option(
        "configs/epic5_active.yaml", "--config", "-c",
        help="Epic 5 YAML config",
    ),
    n_forward: int = typer.Option(
        20, "--n-forward", help="Number of MC Dropout forward passes",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output CSV path (default: <output.dir>/uncertainty.csv)",
    ),
):
    """US 5.1 — Run MC Dropout uncertainty estimation on the target domain."""
    import torch

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic5_active"))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output or (out_dir / "uncertainty.csv")

    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)
    batch_size = cfg.get("uncertainty", {}).get("batch_size", 4)

    # Load model
    from udm_epic4.models.dann import DANNModel

    model_cfg = cfg.get("model", {})
    model = DANNModel(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        pretrained=False,
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32]),
        domain_head_hidden=model_cfg.get("domain_head_hidden", 256),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.to(device)

    # Build target dataset
    from udm_epic4.data.multi_domain_dataset import DomainDataset

    tgt_cfg = cfg.get("data", {}).get("target_unlabeled", {})
    target_ds = DomainDataset(
        tgt_cfg["images"], masks_dir=None, domain_label=1, image_size=(h, w),
    )

    # Run MC Dropout uncertainty
    from udm_epic5.uncertainty.mc_dropout import mc_dropout_uncertainty

    rprint(f"[bold]Running MC Dropout[/] (T={n_forward}) on {len(target_ds)} images ...")
    df = mc_dropout_uncertainty(
        model, target_ds, n_forward=n_forward, device=device, batch_size=batch_size,
    )
    df.to_csv(output_csv, index=False)
    rprint(f"[bold green]Done[/] Uncertainty scores saved: [cyan]{output_csv}[/]")
    rprint(f"  Mean entropy: {df['mean_entropy'].mean():.4f}")
    rprint(f"  Max entropy:  {df['mean_entropy'].max():.4f}")


# ======================================================================
# US 5.2 / 5.3 — Active sample selection
# ======================================================================


@app.command("select")
def select(
    uncertainty_csv: Path = typer.Option(
        ..., "--uncertainty-csv", help="CSV from the uncertainty command",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", help="Model checkpoint (needed for feature extraction)",
    ),
    config: Path = typer.Option(
        "configs/epic5_active.yaml", "--config", "-c",
        help="Epic 5 YAML config",
    ),
    budget: int = typer.Option(50, "--budget", "-b", help="Number of samples to select"),
    alpha: float = typer.Option(0.7, "--alpha", help="Uncertainty vs diversity weight"),
    strategy: str = typer.Option(
        "combined", "--strategy", "-s",
        help="Selection strategy: uncertainty, diversity, combined",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output CSV path (default: <output.dir>/selected.csv)",
    ),
):
    """US 5.2/5.3 — Select the most informative target samples for labeling."""
    import numpy as np
    import pandas as pd

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic5_active"))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output or (out_dir / "selected.csv")

    unc_df = pd.read_csv(uncertainty_csv)
    scores = unc_df["mean_entropy"].values

    rprint(f"[bold]Strategy:[/] {strategy}  |  Budget: {budget}  |  Alpha: {alpha}")

    if strategy == "uncertainty":
        # Pure uncertainty: pick top-k by entropy
        order = np.argsort(-scores)
        selected_indices = order[:budget]
    elif strategy == "diversity":
        # Need features — extract from model
        features = _extract_features(checkpoint, config, cfg, unc_df)
        from udm_epic5.selection.diversity import coreset_selection
        selected_indices = coreset_selection(features, budget=budget)
    else:
        # Combined (default)
        features = _extract_features(checkpoint, config, cfg, unc_df)
        from udm_epic5.selection.combined import combined_selection
        selected_indices = combined_selection(
            scores, features, budget=budget, alpha=alpha,
        )

    # Export
    from udm_epic5.selection.combined import export_selection_csv

    export_selection_csv(selected_indices, unc_df, str(output_csv))
    rprint(f"[bold green]Done[/] Selected {len(selected_indices)} samples: [cyan]{output_csv}[/]")


def _extract_features(
    checkpoint: Optional[Path],
    config: Path,
    cfg: dict,
    unc_df,
) -> "np.ndarray":
    """Extract bottleneck features from the model for diversity selection."""
    import numpy as np
    import torch

    if checkpoint is None:
        # Fallback: use random features (user warned)
        rprint("[yellow]Warning:[/] No checkpoint provided — using random features for diversity")
        return np.random.randn(len(unc_df), 128).astype(np.float32)

    img_cfg = cfg.get("image", {})
    h, w = img_cfg.get("height", 512), img_cfg.get("width", 512)

    from udm_epic4.models.dann import DANNModel
    from udm_epic4.data.multi_domain_dataset import DomainDataset

    model_cfg = cfg.get("model", {})
    model = DANNModel(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        pretrained=False,
        decoder_channels=model_cfg.get("decoder_channels", [256, 128, 64, 32]),
        domain_head_hidden=model_cfg.get("domain_head_hidden", 256),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.to(device).eval()

    tgt_cfg = cfg.get("data", {}).get("target_unlabeled", {})
    target_ds = DomainDataset(
        tgt_cfg["images"], masks_dir=None, domain_label=1, image_size=(h, w),
    )

    # Extract bottleneck features
    from torch.utils.data import DataLoader

    loader = DataLoader(target_ds, batch_size=4, shuffle=False)
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feat = model.encode(images)  # [B, C, H', W']
            feat = feat.mean(dim=[2, 3])  # global average pool -> [B, C]
            all_feats.append(feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


# ======================================================================
# US 5.4 — Labeling session preparation
# ======================================================================


@app.command("prepare-session")
def prepare_session(
    selected_csv: Path = typer.Option(
        ..., "--selected-csv", help="CSV from the select command",
    ),
    images_dir: Path = typer.Option(
        ..., "--images-dir", help="Root directory of target-domain images",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Session output directory",
    ),
    session_name: str = typer.Option(
        "", "--session-name", help="Human-friendly session name",
    ),
):
    """US 5.4 — Create a labeling session folder with selected images."""
    from udm_epic5.labeling.session import LabelingSession

    session_root = output or Path("labeling")
    name = session_name or "session_001"

    session = LabelingSession(
        selected_csv=str(selected_csv),
        images_dir=str(images_dir),
        output_dir=str(session_root),
        session_name=name,
    )
    session.prepare()

    status = session.status()
    rprint(f"[bold green]Done[/] Labeling session created: [cyan]{session.session_dir}[/]")
    rprint(f"  Images: {status['total']}")
    rprint(f"  Labeled: {status['labeled']}  |  Unlabeled: {status['unlabeled']}")
    rprint("\n[bold]Next steps:[/]")
    rprint(f"  1. Annotate masks in: [cyan]{session.masks_out}[/]")
    rprint("  2. Name masks to match image stems (e.g. img001.png)")
    rprint("  3. Run: udm-epic5 train --config configs/epic5_active.yaml")


# ======================================================================
# US 5.5 — Active DANN training
# ======================================================================


@app.command("train")
def train(
    config: Path = typer.Option(
        "configs/epic5_active.yaml", "--config", "-c",
        help="Active DANN training config",
    ),
):
    """US 5.5 — Train Active DANN with labeled target samples."""
    from udm_epic5.active_training import train_active_dann_from_yaml

    rprint("[bold]Starting Active DANN training ...[/]")
    best_ckpt = train_active_dann_from_yaml(config)
    rprint(f"[bold green]Done[/] Best checkpoint: [cyan]{best_ckpt}[/]")


# ======================================================================
# US 5.6 — Learning curve analysis
# ======================================================================


@app.command("analyze")
def analyze(
    results_csv: Path = typer.Option(
        ..., "--results-csv", help="CSV with per-round evaluation results",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory for plots (default: same dir as CSV)",
    ),
):
    """US 5.6 — Plot learning curves and compare active-learning strategies."""
    from udm_epic5.analysis.learning_curve import (
        build_learning_curve_df,
        check_stopping_criterion,
        plot_learning_curves,
    )

    df = build_learning_curve_df(results_csv)
    save_dir = output or results_csv.parent
    paths = plot_learning_curves(df, save_dir=save_dir)

    rprint(f"[bold green]Done[/] Plots saved:")
    for p in paths:
        rprint(f"  [cyan]{p}[/]")

    # Check stopping criterion
    stop = check_stopping_criterion(df, metric="dice", threshold=0.02)
    if stop:
        rprint("[yellow]Stopping criterion met[/] — diminishing returns from more labeling")
    else:
        rprint("[green]Continue[/] — improvement still above threshold")


# ======================================================================
# Convenience: run one complete active learning round
# ======================================================================


@app.command("run-round")
def run_round(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Model checkpoint (.pth)",
    ),
    config: Path = typer.Option(
        "configs/epic5_active.yaml", "--config", "-c",
        help="Epic 5 YAML config",
    ),
    budget: int = typer.Option(50, "--budget", "-b", help="Samples to select"),
    round_num: int = typer.Option(1, "--round-num", "-r", help="Round number"),
):
    """Run one complete active learning round: uncertainty -> select -> prepare-session."""
    import pandas as pd

    cfg = _load_yaml(config)
    out_dir = Path(cfg.get("output", {}).get("dir", "outputs/epic5_active"))
    round_dir = out_dir / f"round_{round_num:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Uncertainty
    rprint(f"\n[bold]=== Round {round_num} — Step 1/3: Uncertainty ===[/]")
    unc_csv = round_dir / "uncertainty.csv"
    uncertainty(
        checkpoint=checkpoint,
        config=config,
        n_forward=cfg.get("uncertainty", {}).get("n_forward", 20),
        output=unc_csv,
    )

    # Step 2: Selection
    rprint(f"\n[bold]=== Round {round_num} — Step 2/3: Selection ===[/]")
    sel_csv = round_dir / "selected.csv"
    sel_cfg = cfg.get("selection", {})
    select(
        uncertainty_csv=unc_csv,
        checkpoint=checkpoint,
        config=config,
        budget=budget,
        alpha=sel_cfg.get("alpha", 0.7),
        strategy=sel_cfg.get("strategy", "combined"),
        output=sel_csv,
    )

    # Step 3: Prepare labeling session
    rprint(f"\n[bold]=== Round {round_num} — Step 3/3: Prepare Session ===[/]")
    tgt_cfg = cfg.get("data", {}).get("target_unlabeled", {})
    images_dir = tgt_cfg.get("images", "data/malaysia/images")
    labeling_dir = round_dir / "labeling"
    prepare_session(
        selected_csv=sel_csv,
        images_dir=Path(images_dir),
        output=labeling_dir,
        session_name=f"round_{round_num:02d}",
    )

    rprint(f"\n[bold green]Round {round_num} complete.[/]")
    rprint("After labeling, run: udm-epic5 train --config configs/epic5_active.yaml")


if __name__ == "__main__":
    app()
