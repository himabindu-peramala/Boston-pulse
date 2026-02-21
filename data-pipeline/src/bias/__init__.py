"""
Boston Pulse - Bias Detection System

Fairness and bias detection components for ensuring equitable data:
- Data slicing by demographic and geographic dimensions
- Fairness metrics computation
- Model card generation

IMPORTANT: Boston Pulse deals with public safety data. Bias in this data
can perpetuate systemic inequities. These checks are CRITICAL.

Components:
    - DataSlicer: Categorical, quantile, range slicing
    - FairnessChecker: Slice-based evaluation, FairnessGate blocker
    - ModelCardGenerator: Markdown + JSON model cards
"""

from src.bias.data_slicer import DataSlice, DataSlicer, slice_data
from src.bias.fairness_checker import (
    FairnessChecker,
    FairnessResult,
    FairnessSeverity,
    FairnessViolationError,
    check_fairness,
)
from src.bias.model_card_generator import (
    ModelCard,
    ModelCardGenerator,
    create_model_card,
)

__version__ = "0.1.0"

__all__ = [
    # Data Slicer
    "DataSlicer",
    "DataSlice",
    "slice_data",
    # Fairness Checker
    "FairnessChecker",
    "FairnessResult",
    "FairnessSeverity",
    "FairnessViolationError",
    "check_fairness",
    # Model Card Generator
    "ModelCardGenerator",
    "ModelCard",
    "create_model_card",
]
