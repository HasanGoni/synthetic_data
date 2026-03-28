"""Reporting modules for Epic 4 DANN.

Provides failure categorisation, visual diagnostics, and structured
reporting for post-training analysis of domain-adaptive segmentation.
"""

from .failure_analysis import categorize_failures, generate_failure_report

__all__ = [
    "categorize_failures",
    "generate_failure_report",
]
