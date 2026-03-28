"""Labeling session management for Epic 5 — human-in-the-loop workflow.

Provides tools to prepare labeling sessions from active-learning selections,
track annotation progress, and load completed labels for training.
"""

from .session import LabelingSession, create_labeling_session, load_labeled_samples

__all__ = [
    "LabelingSession",
    "create_labeling_session",
    "load_labeled_samples",
]
