"""Evaluation modules for Epic 4 DANN.

Provides segmentation metrics, model evaluation across domains, t-SNE
visualisation, and domain confusion analysis for assessing the quality
of domain-adversarial training.
"""

from .domain_analysis import (
    compute_tsne,
    domain_confusion_score,
    extract_features,
    plot_tsne,
)
from .metrics import (
    compute_dice,
    compute_f1,
    compute_iou,
    evaluate_all_domains,
    evaluate_model_on_domain,
)

__all__ = [
    # metrics
    "compute_f1",
    "compute_iou",
    "compute_dice",
    "evaluate_model_on_domain",
    "evaluate_all_domains",
    # domain analysis
    "extract_features",
    "compute_tsne",
    "plot_tsne",
    "domain_confusion_score",
]
