"""
UDM Epic 4 — Domain Adversarial Neural Networks (DANN) for multi-site deployment.

Train domain-invariant defect segmentation models that work across
manufacturing sites (Warstein, Malaysia, Regensburg, China) without
target domain labels.

Install extras: ``pip install -e ".[epic4]"``
"""

from udm_epic4.training.lambda_scheduler import dann_lambda_schedule

__all__ = [
    "dann_lambda_schedule",
]
