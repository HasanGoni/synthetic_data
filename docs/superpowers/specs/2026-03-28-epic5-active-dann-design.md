# Epic 5: Active Domain Adaptation — Design Spec

**Date:** 2026-03-28
**Author:** Hasan Goni
**Status:** Approved
**Notion:** [Epic 5](https://www.notion.so/303ea36d4bc081978898e0622d35ae23)
**Depends on:** Epic 4 (DANN)

---

## Goal

Intelligently select the most informative target domain samples to label,
achieving 85%+ target F1 with only 50 labeled samples (10% of labeling budget).
Builds on Epic 4's DANN as the base model.

**Success criteria:**
- 85%+ Malaysia F1 with 50 labeled target samples
- 91% reduction in labeling effort (50 vs 500 samples)
- Works with DANN from Epic 4
- Multi-round iteration with convergence detection

---

## Architecture

```
Train source model (Epic 4 DANN)
        |
        v
Run MC Dropout on target (T=20 passes)
        |
        v
Compute uncertainty + diversity scores
        |
        v
Select top-K samples (budget=50)
        |
        v
Human labels selected samples
        |
        v
Retrain: Active DANN (source + labeled target + unlabeled target)
        |
        v
Iterate 2-3 rounds until convergence
```

### Key Components

**Uncertainty (MC Dropout):**
- T=20 stochastic forward passes with dropout enabled
- Per-pixel entropy from probability ensemble
- Image-level: mean entropy + max entropy + prediction variance

**Diversity (Coreset):**
- Extract bottleneck features via model.encode()
- Greedy farthest-first traversal for maximum coverage
- Alternative: k-means clustering selection

**Combined Selection:**
- Score = alpha * uncertainty + (1-alpha) * diversity
- Default alpha=0.7 (uncertainty-weighted)
- Greedy iterative: pick best score, update diversity distances

**Active DANN Training:**
- Modified DANN: segmentation loss on source + labeled target
- L = L_seg(source + labeled_target) + lambda * L_domain(all)
- Fine-tune from Epic 4 checkpoint with lower learning rate

---

## Package Structure

```
udm_epic5/
    __init__.py
    cli_epic5.py                      # 6 CLI commands
    uncertainty/
        __init__.py
        mc_dropout.py                 # MC Dropout uncertainty estimation
    selection/
        __init__.py
        diversity.py                  # Coreset + clustering selection
        combined.py                   # Combined uncertainty + diversity
    active_training/
        __init__.py
        train_active_dann.py          # Active DANN training loop
    labeling/
        __init__.py
        session.py                    # Human-in-the-loop session management
    analysis/
        __init__.py
        convergence.py                # Learning curves, strategy comparison
```

---

## CLI Commands

| Command | User Story | Description |
|---------|-----------|-------------|
| `udm-epic5 uncertainty` | US 5.1 | Run MC Dropout on target, output uncertainty CSV |
| `udm-epic5 select` | US 5.2/5.3 | Select top-K samples (uncertainty/diversity/combined) |
| `udm-epic5 prepare-session` | US 5.4 | Create labeling session folder for human annotation |
| `udm-epic5 train` | US 5.5 | Train Active DANN with labeled target samples |
| `udm-epic5 analyze` | US 5.6 | Plot learning curves, compare strategies |
| `udm-epic5 run-round` | Convenience | Chain: uncertainty → select → prepare session |

---

## Expected Results

| Strategy | F1 (50 labels) | F1 (100 labels) | Efficiency |
|----------|----------------|-----------------|------------|
| Random | 70% | 78% | Baseline |
| Uncertainty only | 80% | 85% | 1.4x |
| Diversity only | 75% | 82% | 1.2x |
| **Combined** | **85%** | **90%** | **2.1x** |
| DANN (0 labels) | 78% | 78% | Infinite |
| **Active DANN** | **88%** | **92%** | **2.5x** |
