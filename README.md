# Synthetic Data Generation for Semiconductor Defect Detection

**Universal Detection Model (UDM) — Infineon IEG**

Complete synthetic data generation pipeline covering 9 epics: physics-based synthesis, GAN generation, cross-modality translation, domain adaptation, active learning, bond wire synthesis, spectral imaging, unified pipeline, and crack generation.

**243 tests | 9 CLIs | 48 notebooks | ~31K LOC**

---

## Epic Index

Full details: [`docs/EPIC_INDEX.md`](docs/EPIC_INDEX.md)

| Epic | Name | What it does | CLI |
|------|------|-------------|-----|
| 1 | X-ray Void Synthesis | Physics-based void generation (Beer-Lambert) | `udm-generate` |
| 2 | GAN-Based Defect Gen | ControlNet + Stable Diffusion fine-tuning | `udm-epic2` |
| 3 | CycleGAN Translation | AOI ↔ USM cross-modality image translation | `udm-epic3` |
| 4 | DANN Multi-Site | Domain adaptation across 4 manufacturing sites | `udm-epic4` |
| 5 | Active Learning | Smart sample selection (MC Dropout + Coreset) | `udm-epic5` |
| 6 | Bond Wire Synthesis | AOI bond wire defect generation | `udm-epic6` |
| 7 | Chromasense Spectral | Multi-wavelength spectral image synthesis | `udm-epic7` |
| 8 | Universal Pipeline | Unified generation + COCO/YOLO/HF export | `udm-epic8` |
| 9 | Crack Generation | Fractal cracks + USM→RGB domain transfer | `udm-epic9` |

---

## Where to Start

### I want to generate synthetic data right now

```bash
# Install
uv sync

# Generate 3000 X-ray void images (Epic 1, CPU only, ~3 min)
udm-generate run --config configs/default.yaml

# Or generate crack defect images (Epic 9)
udm-epic9 generate --config configs/epic9_crack.yaml --domain both
```

### I want to understand how everything works

Start with the **tutorial notebook** — it walks through every function in Epic 1 and Epic 2 with visual output:

```bash
jupyter lab nbs/100_tutorial.ipynb
```

### I want to understand a specific epic

Each epic has numbered notebooks that explain every module step by step:

| Start here | What you learn |
|------------|---------------|
| `nbs/100_tutorial.ipynb` | Complete walkthrough of Epic 1 + 2 (recommended first) |
| `nbs/epic3_00_overview.ipynb` | CycleGAN cross-modality translation |
| `nbs/epic4_00_overview.ipynb` | DANN domain adaptation architecture |
| `nbs/epic5_00_overview.ipynb` | Active learning sample selection |
| `nbs/epic6_00_overview.ipynb` | Bond wire synthesis |
| `nbs/epic7_00_overview.ipynb` | Chromasense spectral imaging |
| `nbs/epic8_00_overview.ipynb` | Universal pipeline + dataset export |
| `nbs/epic9_00_overview.ipynb` | Crack generation + domain transfer |

Each `_00_overview` notebook explains the architecture, then `_01`, `_02`, etc. dive into each module individually with runnable code.

### I want to understand a specific script/module

Every Python module has a corresponding notebook that explains it:

| Module | Explained in |
|--------|-------------|
| `udm_epic1/physics/beer_lambert.py` | `nbs/01_physics.ipynb` |
| `udm_epic1/generators/void_shapes.py` | `nbs/02_void_shapes.ipynb` |
| `udm_epic1/generators/sample_generator.py` | `nbs/03_generator.ipynb` |
| `udm_epic1/dataset/pipeline.py` | `nbs/04_pipeline.ipynb` |
| `udm_epic1/augmentation/transforms.py` | `nbs/05_augmentation.ipynb` |
| `udm_epic2/dataset/crops.py` | `nbs/epic2_01_dataset_prep.ipynb` |
| `udm_epic3/models/cyclegan.py` | `nbs/epic3_02_training.ipynb` |
| `udm_epic4/models/dann.py` | `nbs/epic4_03_dann_training.ipynb` |
| `udm_epic5/uncertainty/mc_dropout.py` | `nbs/epic5_01_uncertainty.ipynb` |
| `udm_epic9/models/crack_geometry.py` | `nbs/epic9_01_crack_generation.ipynb` |
| `udm_epic9/domain_transfer/usm_to_rgb.py` | `nbs/epic9_02_domain_transfer.ipynb` |

---

## Setup

```bash
cd synthetic_data
uv sync

# Optional extras for specific epics:
uv pip install -e ".[epic2]"    # ControlNet (diffusers, accelerate)
uv pip install -e ".[epic4]"    # DANN (segmentation-models-pytorch)
uv pip install -e ".[epic5]"    # Active learning (scikit-learn)
```

---

## Quick Start by Task

### Generate synthetic X-ray voids (Epic 1)

```bash
udm-generate run --config configs/default.yaml
udm-generate preview --n 8
udm-generate validate --output data/synthetic
udm-generate stats --output data/synthetic
```

### Use your own real images as backgrounds (Epic 1)

```python
from udm_epic1 import SyntheticSampleGenerator, GeneratorConfig
import cv2

bg = cv2.imread("your_normal_image.png", cv2.IMREAD_UNCHANGED)
gen = SyntheticSampleGenerator(GeneratorConfig(), seed=42)
image, mask, meta = gen.generate(image_id="sample_001", background=bg)
```

### Train DANN for multi-site deployment (Epic 4)

```bash
udm-epic4 prepare --config configs/epic4_data.yaml
udm-epic4 baseline --config configs/epic4_baseline.yaml
udm-epic4 train --config configs/epic4_dann.yaml
udm-epic4 evaluate --checkpoint outputs/epic4_dann/best.pth --config configs/epic4_evaluate.yaml
```

### Active learning — select samples to label (Epic 5)

```bash
udm-epic5 uncertainty --checkpoint outputs/epic4_dann/best.pth --config configs/epic5_active.yaml
udm-epic5 select --uncertainty-csv outputs/epic5_active/uncertainty.csv --budget 50
udm-epic5 prepare-session --selected-csv outputs/selected.csv --images-dir data/malaysia/images
udm-epic5 train --config configs/epic5_active.yaml
```

### Generate crack defects (Epic 9)

```bash
# Generate full dataset (USM + RGB)
udm-epic9 generate --config configs/epic9_crack.yaml --domain both

# Generate image from mask only
udm-epic9 from-mask --mask my_crack_mask.png --domain rgb --output result.png

# Domain transfer USM → RGB
udm-epic9 transfer --input-dir usm_cracks/ --output-dir rgb_cracks/
```

### Translate between modalities (Epic 3)

```bash
udm-epic3 train --config configs/epic3_cyclegan.yaml
udm-epic3 translate --checkpoint outputs/epic3_cyclegan/latest.pth --input-dir data/aoi/images --output-dir data/translated_usm --direction a2b
```

### Export to COCO/YOLO format (Epic 8)

```bash
udm-epic8 generate --config configs/epic8_universal.yaml
udm-epic8 export --format coco --input data/universal --output data/coco_export
```

---

## Test

```bash
# Run all 243 tests
pytest tests/ -v

# Run tests for a specific epic
pytest tests/test_epic1.py -v
pytest tests/test_epic4.py -v
pytest tests/test_epic9.py -v
```

---

## Project Structure

```
synthetic_data/
├── udm_epic1/          # Physics-based void synthesis (Beer-Lambert)
├── udm_epic2/          # ControlNet defect generation
├── udm_epic3/          # CycleGAN AOI↔USM translation
├── udm_epic4/          # DANN domain adaptation
├── udm_epic5/          # Active learning + MC Dropout
├── udm_epic6/          # Bond wire synthesis (AOI)
├── udm_epic7/          # Chromasense spectral imaging
├── udm_epic8/          # Universal pipeline + export
├── udm_epic9/          # Crack generation + domain transfer
├── configs/            # YAML configs for all epics
├── tests/              # 243 tests (test_epic1.py through test_epic9.py)
├── nbs/                # 48 Jupyter notebooks (tutorials + per-epic guides)
├── docs/
│   ├── EPIC_INDEX.md   # Complete epic reference
│   └── superpowers/specs/  # Design specs for each epic
└── scripts/            # Standalone utility scripts
```

---

## Design Specs

Detailed architecture decisions for each epic:

- [`docs/superpowers/specs/2026-03-28-epic4-dann-design.md`](docs/superpowers/specs/2026-03-28-epic4-dann-design.md)
- [`docs/superpowers/specs/2026-03-28-epic5-active-dann-design.md`](docs/superpowers/specs/2026-03-28-epic5-active-dann-design.md)
- [`docs/superpowers/specs/2026-03-28-epic3-cyclegan-design.md`](docs/superpowers/specs/2026-03-28-epic3-cyclegan-design.md)
- [`docs/superpowers/specs/2026-03-28-epic678-future-design.md`](docs/superpowers/specs/2026-03-28-epic678-future-design.md)
