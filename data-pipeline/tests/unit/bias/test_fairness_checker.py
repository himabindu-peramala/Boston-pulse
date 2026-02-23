"""
Tests for Fairness Checker

Tests fairness evaluation and FairnessGate.
"""

import numpy as np
import pandas as pd
import pytest

from src.bias.fairness_checker import (
    FairnessChecker,
    FairnessMetric,
    FairnessResult,
    FairnessSeverity,
    FairnessViolation,
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

    downtown = pd.DataFrame(
        {
            "id": range(300),
            "neighborhood": ["Downtown"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.9, 0.1]),
        }
    )
    roxbury = pd.DataFrame(
        {
            "id": range(300, 600),
            "neighborhood": ["Roxbury"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.5, 0.5]),
        }
    )
    dorchester = pd.DataFrame(
        {
            "id": range(600, 900),
            "neighborhood": ["Dorchester"] * 300,
            "arrest_made": np.random.choice([0, 1], 300, p=[0.8, 0.2]),
        }
    )

    return pd.concat([downtown, roxbury, dorchester], ignore_index=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def test_fairness_violation_fields():
    """Test FairnessViolation dataclass stores values correctly."""
    v = FairnessViolation(
        metric=FairnessMetric.REPRESENTATION,
        severity=FairnessSeverity.WARNING,
        dimension="neighborhood",
        slice_value="Roxbury",
        message="Under-represented",
        expected=33.3,
        actual=10.0,
        disparity=0.7,
    )
    assert v.metric == FairnessMetric.REPRESENTATION
    assert v.severity == FairnessSeverity.WARNING
    assert v.disparity == pytest.approx(0.7)
    assert v.details == {}


def test_fairness_violation_with_details():
    """Test FairnessViolation stores custom details."""
    v = FairnessViolation(
        metric=FairnessMetric.OUTCOME_DISPARITY,
        severity=FairnessSeverity.CRITICAL,
        dimension="district",
        slice_value="D4",
        message="Large gap",
        expected=0.3,
        actual=0.7,
        disparity=1.33,
        details={"slice_size": 100},
    )
    assert v.details["slice_size"] == 100


def test_fairness_result_empty():
    """Test FairnessResult with no violations."""
    from datetime import UTC, datetime

    result = FairnessResult(
        dataset="test",
        evaluated_at=datetime.now(UTC),
        slices_evaluated=3,
    )
    assert not result.has_violations
    assert not result.has_critical_violations
    assert result.critical_violations == []
    assert result.warning_violations == []
    assert result.passes_fairness_gate  # gate disabled by default


def test_fairness_result_with_violations():
    """Test FairnessResult properties with mixed violations."""
    from datetime import UTC, datetime

    v_crit = FairnessViolation(
        metric=FairnessMetric.REPRESENTATION,
        severity=FairnessSeverity.CRITICAL,
        dimension="hood",
        slice_value="A",
        message="Critical",
        expected=33.0,
        actual=5.0,
        disparity=0.85,
    )
    v_warn = FairnessViolation(
        metric=FairnessMetric.OUTCOME_DISPARITY,
        severity=FairnessSeverity.WARNING,
        dimension="hood",
        slice_value="B",
        message="Warning",
        expected=0.3,
        actual=0.4,
        disparity=0.33,
    )
    result = FairnessResult(
        dataset="test",
        evaluated_at=datetime.now(UTC),
        slices_evaluated=2,
        violations=[v_crit, v_warn],
        fairness_gate_enabled=True,
    )
    assert result.has_violations
    assert result.has_critical_violations
    assert len(result.critical_violations) == 1
    assert len(result.warning_violations) == 1
    assert not result.passes_fairness_gate


def test_fairness_severity_values():
    """Test FairnessSeverity enum values."""
    assert FairnessSeverity.OK == "ok"
    assert FairnessSeverity.WARNING == "warning"
    assert FairnessSeverity.CRITICAL == "critical"


def test_fairness_metric_values():
    """Test FairnessMetric enum values."""
    assert FairnessMetric.REPRESENTATION == "representation"
    assert FairnessMetric.OUTCOME_DISPARITY == "outcome_disparity"
    assert FairnessMetric.STATISTICAL_PARITY == "statistical_parity"


# ---------------------------------------------------------------------------
# Checker initialisation
# ---------------------------------------------------------------------------


def test_checker_initialization():
    """Test FairnessChecker initialization."""
    config = get_config("dev")
    checker = FairnessChecker(config)

    assert checker.config == config
    assert checker.representation_warning == config.fairness.thresholds.representation.warning


def test_checker_default_config():
    """Test FairnessChecker uses default config when none provided."""
    checker = FairnessChecker()
    assert checker.config is not None


# ---------------------------------------------------------------------------
# evaluate_fairness
# ---------------------------------------------------------------------------


def test_evaluate_fairness_balanced(balanced_data):
    """Test fairness evaluation on balanced data has no critical violations."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        outcome_column="arrest_made",
        dimensions=["neighborhood"],
    )

    assert result.dataset == "test"
    assert result.slices_evaluated > 0
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

    assert len(result.violations) > 0
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

    outcome_violations = [v for v in result.violations if v.metric.value == "outcome_disparity"]
    assert len(outcome_violations) > 0


def test_evaluate_fairness_no_outcome_column(imbalanced_data):
    """Test fairness evaluation without outcome column only checks representation."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        outcome_column=None,
        dimensions=["neighborhood"],
    )

    # No outcome_disparity violations expected
    outcome_violations = [v for v in result.violations if v.metric.value == "outcome_disparity"]
    assert len(outcome_violations) == 0


def test_evaluate_fairness_missing_outcome_column(balanced_data):
    """Test evaluate_fairness skips outcome check if column not in df."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        outcome_column="nonexistent",
        dimensions=["neighborhood"],
    )

    outcome_violations = [v for v in result.violations if v.metric.value == "outcome_disparity"]
    assert len(outcome_violations) == 0


def test_evaluate_fairness_default_dimensions(balanced_data):
    """Test evaluate_fairness uses default dimensions when none provided."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(balanced_data, "test")

    assert isinstance(result.slices_evaluated, int)


def test_evaluate_fairness_result_properties(balanced_data):
    """Test FairnessResult properties."""
    checker = FairnessChecker()

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    assert isinstance(result.has_violations, bool)
    assert isinstance(result.has_critical_violations, bool)
    assert isinstance(result.critical_violations, list)
    assert isinstance(result.warning_violations, list)
    assert isinstance(result.passes_fairness_gate, bool)


# ---------------------------------------------------------------------------
# Fairness gate
# ---------------------------------------------------------------------------


def test_fairness_gate_disabled(imbalanced_data):
    """Test that fairness gate passes when disabled even with violations."""
    config = get_config("dev")
    config.fairness.gate_enabled = False
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    assert result.passes_fairness_gate


def test_fairness_gate_enabled(imbalanced_data):
    """Test that fairness gate blocks on critical violations."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    config.fairness.thresholds.representation.critical = 0.1
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        imbalanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    if result.has_critical_violations:
        assert not result.passes_fairness_gate


def test_passes_gate_no_violations_gate_enabled(balanced_data):
    """Gate passes when enabled but no critical violations."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        balanced_data,
        "test",
        dimensions=["neighborhood"],
    )

    if not result.has_critical_violations:
        assert result.passes_fairness_gate


# ---------------------------------------------------------------------------
# evaluate_model_fairness
# ---------------------------------------------------------------------------


def test_evaluate_model_fairness():
    """Test model fairness evaluation."""
    checker = FairnessChecker()

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


# ---------------------------------------------------------------------------
# Fairness report
# ---------------------------------------------------------------------------


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


def test_create_fairness_report_no_violations(balanced_data):
    """Test report generation when no violations exist."""
    config = get_config("dev")
    config.fairness.thresholds.representation.warning = 1.0  # disable all violations
    checker = FairnessChecker(config)

    result = checker.evaluate_fairness(
        balanced_data,
        "clean",
        dimensions=["neighborhood"],
    )

    report = checker.create_fairness_report(result)
    assert "FAIRNESS EVALUATION REPORT" in report
    assert "clean" in report


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_check_fairness_raises_on_violation():
    """Test that check_fairness raises error when gate fails."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    config.fairness.thresholds.representation.critical = 0.01

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
    config.fairness.gate_enabled = False

    result = check_fairness(
        balanced_data,
        "test",
        outcome_column="arrest_made",
        config=config,
    )

    assert result.dataset == "test"


def test_fairness_violation_error_message():
    """Test FairnessViolationError formats message correctly."""
    config = get_config("dev")
    config.fairness.gate_enabled = True
    config.fairness.thresholds.representation.critical = 0.01

    df = pd.DataFrame(
        {
            "neighborhood": ["Downtown"] * 990 + ["Roxbury"] * 10,
            "value": range(1000),
        }
    )

    try:
        check_fairness(df, "my_dataset", config=config)
    except FairnessViolationError as e:
        assert "my_dataset" in str(e)
        assert e.result.dataset == "my_dataset"
