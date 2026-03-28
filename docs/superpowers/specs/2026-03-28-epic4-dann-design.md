# Epic 4: Domain Adversarial Neural Networks (DANN) — Design Spec

**Date:** 2026-03-28
**Author:** Hasan Goni
**Status:** Approved
**Notion:** [Epic 4](https://www.notion.so/303ea36d4bc081ffabd5ea2d2db36360)

---

## Goal

Train domain-invariant defect segmentation models that work across manufacturing sites (Warstein, Malaysia, Regensburg, China) without target domain labels. Close 50%+ of the domain gap using adversarial training.

**Success criteria:**
- Source F1 >= 90% (Warstein)
- Target F1 >= 80% (Malaysia, no target labels used during training)
- Domain classifier accuracy < 60% (features are domain-agnostic)

---

## Architecture

```
Input Image (any site)
    |
    v
+-------------------------------+
|  Shared Encoder               |  ConvNeXt-Tiny (default, via timm)
|  (feature extractor)          |  Configurable: ResNet50, DINOv3 ViT
+-------------------------------+
    |                    |
    v                    v (after GRL)
+--------------+   +--------------------+
| U-Net Decoder|   | Domain Classifier  |
| (segmentation)|   | GAP -> FC(512->    |
|              |   |   256->1)          |
+--------------+   +--------------------+
    |                    |
    v                    v
  Void Mask          Source vs Target
  (supervised,       (adversarial,
   source only)       both domains)
```

### Components

**Shared Encoder:**
- Default: ConvNeXt-Tiny pretrained on ImageNet (via timm)
- Configurable via YAML: `resnet50`, `convnext_tiny`, `vit_small_patch14_dinov2`
- Outputs multi-scale feature maps for U-Net skip connections
- Designed for future DINOv3 ViT swap (aligned with Universal Model direction)

**Segmentation Decoder (U-Net):**
- Skip connections from encoder stages
- Channel progression: [256, 128, 64, 32]
- Final: Conv2d(32, 1) -> sigmoid
- Loss: BCE + Dice (handles small void class imbalance)
- Supervised on source domain only

**Domain Classifier:**
- Gradient Reversal Layer (GRL) before the head
- GRL: identity forward, gradient * (-lambda) backward
- Head: GlobalAvgPool -> FC(feat_dim, 256) -> ReLU -> FC(256, 1) -> sigmoid
- Loss: BCE (source=0, target=1)
- Receives features from both source and target domains

**Gradient Reversal Layer (GRL):**
- Forward pass: identity (pass features through)
- Backward pass: multiply gradients by -lambda
- Effect: encoder learns to fool domain classifier -> domain-invariant features

### Training Protocol

**Loss function:**
```
L_total = L_seg(source) + lambda * L_domain(source + target)
```

**Lambda schedule (DANN paper):**
```
lambda = 2 / (1 + exp(-10 * progress)) - 1
```
Where progress = current_epoch / max_epochs (0 to 1).

**Batch composition:**
- Each batch: 50% source images (with masks) + 50% target images (no masks)
- Custom DomainBatchSampler ensures balanced sampling
- Source images contribute to both seg loss and domain loss
- Target images contribute to domain loss only

**Optimization:**
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Gradient clipping: max_norm=1.0
- Mixed precision: fp16 (configurable)
- Epochs: 100 (configurable)

---

## Package Structure

```
udm_epic4/
    __init__.py
    cli_epic4.py                    # Typer CLI, 6 commands
    models/
        __init__.py
        encoder.py                  # timm backbone wrapper, multi-scale features
        decoder.py                  # U-Net segmentation decoder
        domain_classifier.py        # GRL + domain head
        dann.py                     # Full DANN model (composes encoder+decoder+domain)
    data/
        __init__.py
        multi_domain_dataset.py     # PyTorch Dataset, config-driven paths
        domain_sampler.py           # 50/50 source/target batch sampler
    training/
        __init__.py
        train_baseline.py           # Source-only segmentation (US 4.2)
        train_dann.py               # DANN adversarial training loop (US 4.3)
        lambda_scheduler.py         # GRL lambda schedule
    evaluation/
        __init__.py
        metrics.py                  # F1, IoU, Dice per domain
        domain_analysis.py          # t-SNE, domain confusion, feature viz
    reporting/
        __init__.py
        failure_analysis.py         # Failure categorization + report generation
```

---

## Configuration

### Data config (`configs/epic4_data.yaml`)

```yaml
domains:
  source:
    name: warstein
    images: data/warstein/images
    masks: data/warstein/masks
    train_ratio: 0.8
    val_ratio: 0.2
  targets:
    - name: malaysia
      images: data/malaysia/images
      # no masks: unlabeled for DANN training
  evaluation:
    - name: malaysia_eval
      images: data/malaysia_eval/images
      masks: data/malaysia_eval/masks
    - name: regensburg
      images: data/regensburg/images
      masks: data/regensburg/masks
    - name: china
      images: data/china/images
      masks: data/china/masks

image:
  height: 512
  width: 512
  normalize: true    # ImageNet mean/std
```

### Baseline config (`configs/epic4_baseline.yaml`)

```yaml
model:
  backbone: convnext_tiny
  pretrained: true
  decoder_channels: [256, 128, 64, 32]

training:
  max_epochs: 50
  batch_size: 8
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  mixed_precision: true
  seed: 42

loss:
  segmentation: bce_dice

data:
  source:
    name: warstein
    images: data/warstein/images
    masks: data/warstein/masks
    train_ratio: 0.8
    val_ratio: 0.2

output:
  dir: outputs/epic4_baseline
  checkpoint_every: 10
```

### DANN config (`configs/epic4_dann.yaml`)

```yaml
model:
  backbone: convnext_tiny
  pretrained: true
  decoder_channels: [256, 128, 64, 32]
  domain_head_hidden: 256

training:
  max_epochs: 100
  batch_size: 8          # 4 source + 4 target
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  lambda_max: 1.0
  lambda_schedule: dann  # 2/(1+exp(-10p))-1
  gradient_clip: 1.0
  mixed_precision: true
  seed: 42

loss:
  segmentation: bce_dice
  domain: bce
  seg_weight: 1.0
  domain_weight: 1.0

data:
  source:
    name: warstein
    images: data/warstein/images
    masks: data/warstein/masks
    train_ratio: 0.8
    val_ratio: 0.2
  targets:
    - name: malaysia
      images: data/malaysia/images
  evaluation:
    - name: malaysia_eval
      images: data/malaysia_eval/images
      masks: data/malaysia_eval/masks

output:
  dir: outputs/epic4_dann
  checkpoint_every: 10
```

### Evaluate config (`configs/epic4_evaluate.yaml`)

```yaml
model:
  backbone: convnext_tiny
  decoder_channels: [256, 128, 64, 32]
  domain_head_hidden: 256

checkpoint: outputs/epic4_dann/best.pth

evaluation:
  - name: warstein
    images: data/warstein/images
    masks: data/warstein/masks
  - name: malaysia
    images: data/malaysia_eval/images
    masks: data/malaysia_eval/masks
  - name: regensburg
    images: data/regensburg/images
    masks: data/regensburg/masks
  - name: china
    images: data/china/images
    masks: data/china/masks

output:
  dir: outputs/epic4_evaluation
  save_predictions: true
  save_tsne: true

image:
  height: 512
  width: 512
```

---

## CLI Commands

| Command | User Story | Description |
|---------|-----------|-------------|
| `udm-epic4 prepare` | US 4.1 | Analyze domain shift: dataset stats, intensity histograms, t-SNE of pretrained features |
| `udm-epic4 baseline` | US 4.2 | Train source-only segmentation model (no domain adaptation) |
| `udm-epic4 train` | US 4.3 | Train DANN with gradient reversal layer |
| `udm-epic4 analyze` | US 4.4 | Domain confusion analysis: t-SNE, domain classifier accuracy on frozen features |
| `udm-epic4 evaluate` | US 4.5 | Evaluate checkpoint on all domains, compute F1/IoU/Dice per site |
| `udm-epic4 report` | US 4.6 | Generate failure analysis: categorize errors, confusion matrices, visual report |

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `epic4_00_overview.ipynb` | Architecture diagram, GRL explanation, design decisions |
| `epic4_01_data_analysis.ipynb` | US 4.1: Load multi-site data, visualize domain shift, intensity distributions |
| `epic4_02_baseline.ipynb` | US 4.2: Train baseline, show source vs target gap |
| `epic4_03_dann_training.ipynb` | US 4.3: DANN training walkthrough, lambda schedule, loss curves |
| `epic4_04_domain_analysis.ipynb` | US 4.4: t-SNE before/after DANN, domain classifier confusion |
| `epic4_05_deployment.ipynb` | US 4.5: Multi-site evaluation, comparison table |
| `epic4_06_failure_analysis.ipynb` | US 4.6: Failure modes, visual examples, recommendations |

---

## Tests (`tests/test_epic4.py`)

| Test | What it verifies |
|------|-----------------|
| `TestGRL` | Gradient sign flips in backward, lambda scaling correct |
| `TestEncoder` | Forward pass output shapes, backbone swap via config |
| `TestDecoder` | Skip connections, output matches input spatial dims |
| `TestDANN` | End-to-end forward: input image -> (seg_mask, domain_logit) |
| `TestMultiDomainDataset` | Config-driven loading, source has masks, target returns None mask |
| `TestDomainSampler` | 50/50 source/target batch composition |
| `TestLambdaScheduler` | Values at progress=0 (0), 0.5 (~0.73), 1.0 (~1.0) |
| `TestMetrics` | F1, IoU, Dice computation on known inputs |

---

## pyproject.toml Changes

```toml
[project.scripts]
udm-epic4 = "udm_epic4.cli_epic4:app"

[project.optional-dependencies]
epic4 = [
    "segmentation-models-pytorch>=0.3",
]

[tool.hatch.build.targets.wheel]
packages = ["udm_epic1", "udm_epic2", "udm_epic4"]

[tool.pytest.ini_options]
addopts = "--cov=udm_epic1 --cov=udm_epic2 --cov=udm_epic4 --cov-report=term-missing"
```

---

## Dependencies on Existing Epics

- **Epic 1:** Validation methodology (mask stats), synthetic data for augmenting source domain
- **Epic 2:** Can use ControlNet-generated data to augment source training set
- **Downstream:** Epic 5 (Active Domain Adaptation) builds on DANN as baseline

---

## Expected Results

| Method | Source F1 (Warstein) | Target F1 (Malaysia) | Gap |
|--------|---------------------|---------------------|-----|
| Source-only (no adaptation) | 92% | 65% | 27% |
| **DANN (no target labels)** | 90% | **78%+** | **12%** |
| Fine-tune (10% target labels) | 91% | 85% | 6% |
| Full target training (oracle) | - | 93% | - |

**Success = close 50%+ of the domain gap without target labels.**

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Adversarial training collapse | Gradient clipping, careful lambda schedule, monitor domain classifier accuracy |
| Domain shift too large | Start with same package/defect type, expand later |
| Class imbalance (small voids) | BCE + Dice loss, per-class metrics |
| Backbone swap breaks things | Abstract encoder interface, test with multiple backbones in CI |
