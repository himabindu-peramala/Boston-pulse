"""
Boston Pulse ML - Crime Navigate Model Package.

Cross-sectional model for predicting danger_rate per (h3_index, hour_bucket).
Used by Navigate app to score route safety.
"""

from models.crime_navigate.bias_checker import BiasGateError, check_bias, get_slice_summary
from models.crime_navigate.feature_loader import load_features
from models.crime_navigate.publisher import (
    delete_stale_scores,
    get_collection_stats,
    publish_scores,
    verify_publish,
)
from models.crime_navigate.scorer import get_score_statistics, score_all_cells
from models.crime_navigate.target_builder import build_targets
from models.crime_navigate.trainer import load_model, predict, random_split, train_model
from models.crime_navigate.tuner import get_default_params, tune_hyperparams
from models.crime_navigate.validator import ValidationGateError, validate_model

__all__ = [
    # Feature loading
    "load_features",
    # Target building
    "build_targets",
    # Tuning
    "tune_hyperparams",
    "get_default_params",
    # Training
    "train_model",
    "random_split",
    "load_model",
    "predict",
    # Validation
    "validate_model",
    "ValidationGateError",
    # Bias checking
    "check_bias",
    "BiasGateError",
    "get_slice_summary",
    # Scoring
    "score_all_cells",
    "get_score_statistics",
    # Publishing
    "publish_scores",
    "delete_stale_scores",
    "get_collection_stats",
    "verify_publish",
]
