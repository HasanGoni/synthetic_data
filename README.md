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
│   └── cli.py                       # udm-generate CLI
├── nbs/                             # nbdev source notebooks
├── scripts/generate_dataset.py
├── configs/default.yaml
├── data/synthetic/                  # output: train/val/test + manifest.csv
└── tests/test_epic1.py
```

---

## Setup

```bash
cd ~/projects/udm/udm-epic1-synthetic-data
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Generate

```bash
udm-generate run --config configs/default.yaml --total 3000 --workers 8
udm-generate preview --n 8
udm-generate validate
udm-generate stats
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
