# Epic 3: CycleGAN Cross-Modality Translation — Design Spec

**Date:** 2026-03-28
**Author:** Hasan Goni
**Status:** Approved
**Notion:** [Epic 3](https://www.notion.so/303ea36d4bc081a8ad30d247d7a0cbf5)

---

## Goal

Enable cross-modality knowledge transfer: translate AOI images (abundant labels)
to USM appearance (scarce labels) using unpaired image-to-image translation.
This allows training USM defect models using AOI annotations.

**Success criteria:**
- SSIM >= 0.65 between translated and real target modality
- FID < 100
- Defect preservation Dice >= 0.85 on registered pairs
- Downstream F1 >= 80% of real-data baseline

---

## Architecture

```
AOI Image → G_A2B → Fake USM → D_B (real vs fake USM)
                  ↓
            G_B2A → Reconstructed AOI ≈ original AOI (cycle consistency)

USM Image → G_B2A → Fake AOI → D_A (real vs fake AOI)
                  ↓
            G_A2B → Reconstructed USM ≈ original USM (cycle consistency)
```

### Loss Functions
- Adversarial (LSGAN): MSE-based, real=1, fake=0
- Cycle consistency: L1(reconstructed, original), lambda=10.0
- Identity: L1(G_A2B(real_B), real_B), lambda=5.0
- Defect preservation: Dice(mask_real, mask_translated), lambda=1.0

---

## Package Structure

```
udm_epic3/
    __init__.py
    cli_epic3.py                      # 6 CLI commands
    models/
        __init__.py
        generator.py                  # ResNet-based generator (9 blocks)
        discriminator.py              # PatchGAN 70x70
        cyclegan.py                   # Full CycleGAN model
        losses.py                     # All loss functions
    data/
        __init__.py
        unpaired_dataset.py           # Unpaired + paired datasets
        image_pool.py                 # Image buffer for discriminator
    training/
        __init__.py
        train_cyclegan.py             # Full training loop
    translation/
        __init__.py
        translate.py                  # Batch translation
    evaluation/
        __init__.py
        quality_metrics.py            # SSIM, Dice, FID
```

## CLI Commands

| Command | User Story | Description |
|---------|-----------|-------------|
| `udm-epic3 prepare` | US 3.1 | Analyze multi-modal data |
| `udm-epic3 train` | US 3.2/3.3 | Train CycleGAN |
| `udm-epic3 translate` | US 3.4 | Translate images between modalities |
| `udm-epic3 evaluate` | US 3.4 | SSIM, Dice, quality metrics |
| `udm-epic3 downstream` | US 3.5 | Train on translated data |
| `udm-epic3 validate` | US 3.6 | Multi-site generalization |
