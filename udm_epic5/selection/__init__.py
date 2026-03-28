"""Epic 5 — Sample selection strategies for active domain adaptation.

Re-exports the core selection functions so callers can write::

    from udm_epic5.selection import coreset_selection, combined_selection
"""

from udm_epic5.selection.diversity import coreset_selection, clustering_selection
from udm_epic5.selection.combined import combined_selection, export_selection_csv

__all__ = [
    "coreset_selection",
    "clustering_selection",
    "combined_selection",
    "export_selection_csv",
]
