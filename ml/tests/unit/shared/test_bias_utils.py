"""Tests for shared/bias_utils.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.bias_utils import (
    analyze_slice_fairness,
    compute_slice_metrics,
    format_bias_report,
    rmse_metric,
)


class TestRmseMetric:
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert rmse_metric(y, y) == 0.0

    def test_nonzero(self) -> None:
        y = np.array([0.0, 0.0])
        p = np.array([1.0, 1.0])
        assert rmse_metric(y, p) == 1.0


class TestComputeSliceMetrics:
    def test_default_rmse(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.0, 2.0, 3.0, 4.0])
        sf = pd.Series(["a", "a", "b", "b"])
        df = compute_slice_metrics(y, p, sf)
        assert "rmse" in df.columns
        assert len(df) == 2


class TestAnalyzeSliceFairness:
    def test_passes_when_slices_uniform(self) -> None:
        np.random.seed(0)
        y = np.ones(60)
        p = np.ones(60) + np.random.normal(0, 0.01, 60)
        sf = pd.Series(["x"] * 30 + ["y"] * 30)
        out = analyze_slice_fairness(
            y, p, sf, max_deviation=0.5, min_slice_size=10, max_multiplier=5.0
        )
        assert out["passed"] is True
        assert out["overall_rmse"] >= 0
        assert len(out["slice_results"]) == 2

    def test_format_bias_report(self) -> None:
        analysis = {
            "overall_rmse": 1.0,
            "passed": True,
            "slice_results": {"a": {"passed": True, "too_small": False}},
            "worst_slice": "a",
            "worst_deviation": 0.05,
        }
        rep = format_bias_report(analysis, "2024-01-01", "district")
        assert rep["slice_dimension"] == "district"
        assert rep["gate_passed"] is True
        assert rep["n_slices"] == 1
