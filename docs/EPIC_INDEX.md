# Synthetic Data Generation — Epic Index

Complete list of all epics in the Universal Detection Model synthetic data pipeline.

---

## Overview

| Epic | Name | Approach | CLI | Status |
|------|------|----------|-----|--------|
| **1** | X-ray Void Synthesis Foundation | Physics (Beer-Lambert) | `udm-generate` | Complete |
| **2** | GAN-Based Defect Generation | ControlNet + Stable Diffusion | `udm-epic2` | Complete |
| **3** | CycleGAN Cross-Modality Translation | CycleGAN (AOI ↔ USM) | `udm-epic3` | Complete |
| **4** | Domain Adversarial Neural Networks | DANN + GRL | `udm-epic4` | Complete |
| **5** | Active Domain Adaptation | MC Dropout + Coreset + Active DANN | `udm-epic5` | Complete |
| **6** | AOI Bond Wire Synthesis | Bezier geometry + optical rendering | `udm-epic6` | Complete |
| **7** | Chromasense Spectral Integration | Multi-wavelength reflectance model | `udm-epic7` | Complete |
| **8** | Universal Model Support | Unified pipeline + dataset export | `udm-epic8` | Complete |
| **9** | Synthetic Crack Generation | Fractal crack geometry + domain transfer | `udm-epic9` | Complete |

---

## Epic 1: X-ray Void Synthesis Foundation

**Package:** `udm_epic1/` | **Priority:** P0 | **CLI:** `udm-generate`

Physics-based synthetic void generation using the Beer-Lambert law for X-ray attenuation.
Generates 16-bit X-ray images with realistic void defects (ellipse, blob, elongated, cluster)
and binary segmentation masks.

**Key capabilities:**
- Beer-Lambert physics engine with SFT correction
- 4 void morphology types matching IPC-7711 distribution
- Parallel dataset generation (3000 images in ~3 minutes)
- Use your own real images as backgrounds
- Training-time augmentation (geometric, photometric, domain-shift)

**Commands:** `run`, `preview`, `validate`, `stats`

---

## Epic 2: GAN-Based Defect Generation

**Package:** `udm_epic2/` | **Priority:** P1 | **CLI:** `udm-epic2`

ControlNet + Stable Diffusion fine-tuned on semiconductor defect crops.
Generates photorealistic defect patches conditioned on Canny edge maps,
then pastes them onto real backgrounds via Poisson/alpha blending.

**Key capabilities:**
- Defect crop extraction with connected component analysis
- Canny edge conditioning for ControlNet
- Quality filtering (Laplacian variance gate)
- Poisson blending for seamless integration
- HuggingFace export format

**Commands:** `extract-crops`, `export-hf`, `train`, `generate`, `paste`, `ablation-template`

---

## Epic 3: CycleGAN Cross-Modality Translation

**Package:** `udm_epic3/` | **Priority:** P2 | **CLI:** `udm-epic3`

Unpaired image-to-image translation between AOI and USM modalities.
Enables training USM defect models using abundant AOI labels.

**Key capabilities:**
- ResNet generator (9 blocks) with InstanceNorm
- PatchGAN 70x70 discriminator (LSGAN)
- Cycle consistency + identity + defect preservation losses
- Image pool for stable discriminator training
- Linear LR decay scheduling

**Commands:** `prepare`, `train`, `translate`, `evaluate`, `downstream`, `validate`

---

## Epic 4: Domain Adversarial Neural Networks (DANN)

**Package:** `udm_epic4/` | **Priority:** P0 | **CLI:** `udm-epic4`

Train domain-invariant defect detectors that work across manufacturing sites
(Warstein, Malaysia, Regensburg, China) without target domain labels.

**Key capabilities:**
- ConvNeXt-Tiny encoder (configurable: ResNet50, DINOv3 ViT)
- U-Net segmentation decoder with skip connections
- Gradient Reversal Layer (GRL) for adversarial domain alignment
- Config-driven multi-site data loading
- t-SNE visualization and domain confusion analysis
- Failure categorization and visual reporting

**Commands:** `prepare`, `baseline`, `train`, `analyze`, `evaluate`, `report`

---

## Epic 5: Active Domain Adaptation

**Package:** `udm_epic5/` | **Priority:** P1 | **CLI:** `udm-epic5`

Intelligently select the most informative target domain samples to label.
Achieves 85%+ F1 with only 50 labeled samples (91% labeling reduction).

**Key capabilities:**
- MC Dropout uncertainty (T=20 stochastic forward passes)
- Coreset diversity selection (farthest-first traversal)
- Combined scoring: alpha * uncertainty + (1-alpha) * diversity
- Human-in-the-loop labeling session management
- Active DANN: source + labeled target + domain adversarial
- Multi-round convergence with stopping criterion

**Commands:** `uncertainty`, `select`, `prepare-session`, `train`, `analyze`, `run-round`

---

## Epic 6: AOI Bond Wire & Surface Defect Synthesis

**Package:** `udm_epic6/` | **Priority:** P2 | **CLI:** `udm-epic6`

Generates synthetic AOI images with bond wire defects for semiconductor
packaging inspection. Models wire geometry, optical reflections, and
common failure modes.

**Key capabilities:**
- Bezier curve wire path generation
- Defect types: bent wire, broken wire, wire lift-off
- AOI optical rendering (specular + diffuse)
- On-the-fly dataset generation

**Commands:** `generate`, `preview`, `evaluate`, `stats`

---

## Epic 7: Chromasense Multi-Spectral Integration

**Package:** `udm_epic7/` | **Priority:** P3 | **CLI:** `udm-epic7`

Generates synthetic multi-spectral images matching Chromasense equipment.
Models material reflectance at multiple wavelengths to detect defects
invisible in standard imaging.

**Key capabilities:**
- Multi-wavelength reflectance model (450nm-850nm)
- Material spectra: Si, Cu, solder, mold compound, oxidized Cu
- Defect spectra: delamination, contamination, oxidation
- Spectral-to-RGB visualization
- Spectral Angle Mapper (SAM) metrics

**Commands:** `generate`, `preview`, `evaluate`, `visualize-spectrum`

---

## Epic 8: Universal Model Support & Integration

**Package:** `udm_epic8/` | **Priority:** P1 | **CLI:** `udm-epic8`

Unified pipeline combining all previous epics into one cross-modality
synthetic data generation system. Registry-based modality management
with multi-format dataset export.

**Key capabilities:**
- Modality registry (xray, aoi, usm, chromasense, etc.)
- Unified generation pipeline across all modalities
- Dataset export: COCO, YOLO, HuggingFace formats
- Dataset merging across modalities
- Cross-modality evaluation reporting

**Commands:** `generate`, `list-modalities`, `export`, `merge`, `report`

---

## Epic 9: Synthetic Crack Generation

**Package:** `udm_epic9/` | **Priority:** P1 | **CLI:** `udm-epic9`

Generates synthetic crack defect images for USM and RGB domains.
Three capabilities: crack synthesis on USM, domain transfer to RGB,
and mask-to-image generation (input crack mask → output domain image).

**Key capabilities:**
- Fractal crack path generation (midpoint displacement)
- 4 crack types: die crack, substrate crack, mold crack, delamination
- USM rendering with acoustic impedance effects
- USM→RGB domain transfer (colormap + learned)
- Mask-to-image: generate full image from crack mask alone

**Commands:** `generate`, `from-mask`, `transfer`, `preview`, `evaluate`, `stats`

---

## Dependency Graph

```
Epic 1 (Foundation)
    │
    ├── Epic 2 (ControlNet)
    ├── Epic 3 (CycleGAN) ──────────────┐
    ├── Epic 4 (DANN) ──── Epic 5 (Active)  │
    ├── Epic 6 (Bond Wire)               │
    ├── Epic 7 (Chromasense)             │
    └── Epic 9 (Crack Generation) ───────┘
                                          │
                                    Epic 8 (Universal)
```

---

## Quick Start

```bash
# Install
cd synthetic_data
uv sync

# Generate X-ray voids (Epic 1)
udm-generate run --config configs/default.yaml

# Train DANN for multi-site deployment (Epic 4)
udm-epic4 train --config configs/epic4_dann.yaml

# Active learning sample selection (Epic 5)
udm-epic5 uncertainty --checkpoint outputs/epic4_dann/best.pth --config configs/epic5_active.yaml
udm-epic5 select --uncertainty-csv outputs/uncertainty.csv --budget 50

# Generate crack defects (Epic 9)
udm-epic9 generate --config configs/epic9_crack.yaml --domain both

# Generate from mask only
udm-epic9 from-mask --mask my_crack_mask.png --domain rgb --output result.png

# Universal pipeline (Epic 8)
udm-epic8 generate --config configs/epic8_universal.yaml
udm-epic8 export --format coco --output data/coco_export
```
