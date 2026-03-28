"""UDM Epic 5 — Active Domain Adaptation.

Combines uncertainty estimation (MC Dropout), active sample selection
(uncertainty / diversity / combined), labeling session management,
active DANN training, and learning-curve analysis into a single
active-learning loop for domain adaptation.
"""

from udm_epic5.uncertainty.mc_dropout import mc_dropout_uncertainty
from udm_epic5.selection.diversity import coreset_selection
from udm_epic5.selection.combined import combined_selection

__all__ = [
    "mc_dropout_uncertainty",
    "coreset_selection",
    "combined_selection",
]
