"""
Tests for Fairness Checker

Tests fairness evaluation and FairnessGate.
"""

import numpy as np
import pandas as pd
import pytest

from src.bias.fairness_checker import (
    FairnessChecker,
    FairnessViolationError,
    check_fairness,
)
from src.shared.config import get_config


@pytest.fixture
def balanced_data():
    """Dataset with balanced representation."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(900),
            "neighborhood": (["Downtown"] * 300 + ["Roxbury"] * 300 + ["Dorchester"] * 300),
            "arrest_made": np.random.choice([0, 1], 900, p=[0.7, 0.3]),
            "value": np.random.normal(100, 15, 900),
        }
    )


@pytest.fixture
def imbalanced_data():
    """Dataset with imbalanced representation."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(900),
            "neighborhood": (["Downtown"] * 600 + ["Roxbury"] * 200 + ["Dorchester"] * 100),
            "arrest_made": np.random.choice([0, 1], 900, p=[0.7, 0.3]),
            "value": np.random.normal(100, 15, 900),
        }
    )


@pytest.fixture
def outcome_disparity_data():
    """Dataset with outcome disparity."""
    np.random.seed(42)

    # Create data with different arrest rates by neighborhood
    downtown = pd.DataFrame(
        {
            "id": range(300),
            "neighborhood": ["Downtown"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.9, 0.1]),  # 10% arrests
        }
    )

    roxbury = pd.DataFrame(
        {
            "id": range(300, 600),
            "neighborhood": ["Roxbury"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.5, 0.5]),  # 50% arrests
        }
    )

    dorchester = pd.DataFrame(
        {
            "id": range(600, 900),
            "neighborhood": ["Dorchester"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.8, 0.2]),  # 20% arrests
        }
    )

    return pd.concat([downtown, roxbury, dorchester], ignore_index=True)


def test_checker_initialization():
    """Test FairnessChecker initialization."""
    config = get_config("dev")
    checker = FairnessChecker(config)

    assert checker.config == config
    assert checker.representation_warning == config.fairness.thresholds.representation.warning


def test_evaluate_fairness_balanced(balanced_data):
    """Test fairness evaluation on balanced data."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        outcome_column="arrest_made",
        dimensions=["neighborhood"],
    )

    assert result.dataset == "test"
    assert result.slices_evaluated > 0
    # Balanced data should have few violations
    assert len(result.critical_violations) == 0


def test_evaluate_fairness_imbalanced(imbalanced_data):
    """Test fairness evaluation detects imbalanced representation."""
    config = get_config("dev")
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    # Should detect representation imbalance
    assert len(result.violations) > 0
    # Check for representation violations
    rep_violations = [v for v in result.violations if v.metric.value == "representation"]
    assert len(rep_violations) > 0


def test_evaluate_fairness_outcome_disparity(outcome_disparity_data):
    """Test fairness evaluation detects outcome disparity."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        outcome_disparity_data,
        "test",
        outcome_column="arrest_made",
        dimensions=["neighborhood"],
    )

    # Should detect outcome disparity
    outcome_violations = [v for v in result.violations if v.metric.value == "outcome_disparity"]
    assert len(outcome_violations) > 0


def test_fairness_gate_disabled(imbalanced_data):
    """Test that fairness gate can be disabled."""
    config = get_config("dev")
    config.fairness.gate_enabled = False
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    # Gate should pass even with violations when disabled
    assert result.passes_fairness_gate


def test_fairness_gate_enabled(imbalanced_data):
    """Test that fairness gate blocks on critical violations."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    config.fairness.thresholds.representation.critical = 0.1  # Very strict
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    # Should have critical violations with very strict threshold
    if result.has_critical_violations:
        assert not result.passes_fairness_gate


def test_evaluate_model_fairness():
    """Test model fairness evaluation."""
    checker = FairnessChecker()

    # Create data with model predictions
    df = pd.DataFrame(
        {
            "id": range(900),
            "neighborhood": (["Downtown"] * 300 + ["Roxbury"] * 300 + ["Dorchester"] * 300),
            "prediction": np.random.choice([0, 1], 900),
            "gender": np.random.choice(["M", "F"], 900),
        }
    )

    result = checker.evaluate_model_fairness(
        df,
        predictions_column="prediction",
        protected_attributes=["neighborhood", "gender"],
        dataset="test",
    )

    assert result.dataset == "test"
    assert result.slices_evaluated > 0


def test_fairness_result_properties(balanced_data):
    """Test FairnessResult properties."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    # Test properties
    assert isinstance(result.has_violations, bool)
    assert isinstance(result.has_critical_violations, bool)
    assert isinstance(result.critical_violations, list)
    assert isinstance(result.warning_violations, list)
    assert isinstance(result.passes_fairness_gate, bool)


def test_create_fairness_report(imbalanced_data):
    """Test fairness report generation."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    report = checker.create_fairness_report(result)

    assert isinstance(report, str)
    assert "FAIRNESS EVALUATION REPORT" in report
    assert "test" in report


def test_check_fairness_raises_on_violation():
    """Test that check_fairness raises error when gate fails."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    config.fairness.thresholds.representation.critical = 0.01  # Very strict

    # Create highly imbalanced data
    df = pd.DataFrame(
        {
            "neighborhood": ["Downtown"] * 950 + ["Roxbury"] * 50,
            "value": range(1000),
        }
    )

    with pytest.raises(FairnessViolationError) as exc_info:
        check_fairness(df, "test", config=config)

    assert "test" in str(exc_info.value)


def test_check_fairness_convenience_function(balanced_data):
    """Test convenience check_fairness function."""
    config = get_config("dev")
    config.fairness.gate_enabled = False  # Disable gate for this test

    result = check_fairness(
        balanced_data,
        "test",
        outcome_column="arrest_made",
        config=config,
    )

    assert result.dataset == "test"
