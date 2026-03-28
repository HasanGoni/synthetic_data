# UDM Epic 1 — Synthetic X-ray Void Dataset Generator

**Part of the Universal Detection Model (UDM) project — Infineon IEG**

Physics-based synthetic data generation for void detection in X-ray solder joint images.
Uses Beer-Lambert law to simulate realistic void contrast across Nordson MatriX and SAKI 3Xi-M110 machines.

---

## Physics Basis

**Beer-Lambert law:** `I = I₀ · exp(−μ · t)`

| Region | μ (attenuation) | Result in X-ray |
|--------|----------------|-----------------|
| Solder | μ > 0 | Dark (high attenuation) |
| Void (air) | μ ≈ 0 | **Bright** (no attenuation) |

---

## Structure

```
udm-epic1-synthetic-data/
├── udm_epic1/
│   ├── physics/beer_lambert.py      # Beer-Lambert + SFT + noise
│   ├── generators/void_shapes.py    # Ellipse/blob/elongated/cluster
│   ├── generators/sample_generator.py
│   ├── augmentation/transforms.py   # Training-time only (NOT saved)
│   ├── dataset/pipeline.py          # Parallel joblib pipeline
│   ├── validation/mask_stats.py     # Real-mask metrics (used by analysis script)
│   └── cli.py                       # udm-generate CLI
├── udm_epic2/                       # Epic 2 — ControlNet pipeline (US 2.1–2.6)
│   ├── dataset/crops.py, hf_export.py
│   ├── conditioning/edges.py
│   ├── training/                    # ControlNet fine-tuning
│   ├── generation/inference.py
│   ├── integration/paste.py
│   ├── quality/filter.py
│   └── cli_epic2.py                 # udm-epic2
├── nbs/                             # epic1_* = Epic 1; epic2_* = Epic 2
│   ├── epic1_100_tutorial_real_backgrounds.ipynb
│   ├── epic2_00_gan_overview.ipynb
│   └── epic2_01_dataset_prep.ipynb
├── scripts/
│   ├── generate_dataset.py
│   ├── analysis/analyze_real_voids.py
│   ├── validation/compare_distributions.py
│   └── epic2/extract_defect_crops.py  # launcher → udm-epic2
├── configs/default.yaml
├── configs/epic2_dataset.yaml
├── configs/epic2_train.yaml
├── configs/epic2_generate.yaml
├── data/synthetic/                  # Epic 1 output: train/val/test + manifest.csv
└── tests/test_epic1.py, tests/test_epic2.py
```

---

## Setup

```bash
cd ~/projects/udm/udm-epic1-synthetic-data
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# Epic 2 (ControlNet train + generate): also install
uv pip install -e ".[epic2]"
```

## Generate

```bash
udm-generate run --config configs/default.yaml --total 3000 --workers 8
udm-generate preview --n 8
udm-generate validate
udm-generate stats
```

## Epic 2 — full pipeline (Notion: GAN / ControlNet epic)

Install extras: `uv pip install -e ".[epic2]"` (diffusers, accelerate, transformers).

| Step | Command |
|------|---------|
| US 2.1–2.2 Crops + edges | `udm-epic2 extract-crops --image-dir ... --mask-dir ... -o data/epic2/crops -c configs/epic2_dataset.yaml` |
| Optional HF export | `udm-epic2 export-hf --crops-root data/epic2/crops -o data/epic2/hf_export` |
| US 2.3 Train ControlNet | `udm-epic2 train -c configs/epic2_train.yaml` |
| US 2.4 Generate | `udm-epic2 generate -c configs/epic2_generate.yaml` |
| US 2.5 Paste on real BG | `udm-epic2 paste -b bg.png -p patch.png -m mask.png --center-x 256 --center-y 256 -o out.png` |
| US 2.6 Ablation CSV | `udm-epic2 ablation-template -o results/epic2_ablation_template.csv` |

Configs: `configs/epic2_dataset.yaml`, `configs/epic2_train.yaml`, `configs/epic2_generate.yaml`.

## Real masks & validation

```bash
python scripts/analysis/analyze_real_voids.py --mask-dir /path/to/masks --output results/void_statistics.json
python scripts/validation/compare_distributions.py --manifest data/synthetic/manifest.csv --real-stats results/void_statistics.json
```

## Test

```bash
pytest tests/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Beer-Lambert physics | Physically grounded contrast |
| 4 void morphologies | Covers IPC-7711 void phenotype distribution |
| p2/p98 normalization | Baked into ONNX — robust to hot pixels |
| Augmentation NOT saved | Avoids overfitting on augmented artifacts |
| 16-bit PNG output | Preserves full detector dynamic range |
| 15% empty images | Hard negatives — no false positives |
| Domain-shift aug | Ring artifacts + seams → P2 readiness |

---

Covers **P0 (Anchor)** and **P1 (Pilot)** of the UDM Complexity Map.
