"""Tests for ModelRegistry.compare_to_production (Gate 3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shared.registry import ModelRegistry


@pytest.fixture
def registry_cfg(sample_cfg: dict) -> dict:
    """Use sample_cfg from conftest."""
    return sample_cfg


@pytest.fixture
def mock_bucket(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock GCS bucket."""
    bucket = MagicMock()
    client = MagicMock()
    client.bucket.return_value = bucket
    monkeypatch.setattr("shared.registry.storage.Client", lambda: client)
    return bucket


@pytest.fixture
def registry_no_prod(registry_cfg: dict, mock_bucket: MagicMock) -> ModelRegistry:
    """
    Registry with no production model (cold start scenario).
    get_latest_metadata() raises an exception.
    """
    registry_cfg["registry"]["use_artifact_registry"] = False
    reg = ModelRegistry(registry_cfg)

    # Make get_latest_metadata raise to simulate cold start
    def raise_not_found():
        raise Exception("No production model found")

    mock_bucket.blob.return_value.download_as_text.side_effect = raise_not_found
    return reg


@pytest.fixture
def registry_with_prod(registry_cfg: dict, mock_bucket: MagicMock) -> ModelRegistry:
    """
    Registry with existing production model.
    Production model has val_rmse=0.5, version=20260315.
    """
    registry_cfg["registry"]["use_artifact_registry"] = False
    reg = ModelRegistry(registry_cfg)

    # Mock get_latest_metadata to return production model info
    prod_meta = json.dumps(
        {
            "version": "20260315",
            "val_rmse": 0.5,
            "execution_date": "2026-03-15",
        }
    )
    mock_bucket.blob.return_value.download_as_text.return_value = prod_meta
    return reg


@pytest.fixture
def registry_with_prod_no_rmse(registry_cfg: dict, mock_bucket: MagicMock) -> ModelRegistry:
    """
    Registry with existing production model but no val_rmse in metadata.
    This can happen with legacy models.
    """
    registry_cfg["registry"]["use_artifact_registry"] = False
    reg = ModelRegistry(registry_cfg)

    prod_meta = json.dumps(
        {
            "version": "20260315",
            "execution_date": "2026-03-15",
        }
    )
    mock_bucket.blob.return_value.download_as_text.return_value = prod_meta
    return reg


class TestCompareToProductionColdStart:
    """Tests for cold start scenario (no production model)."""

    def test_cold_start_should_promote(self, registry_no_prod: ModelRegistry) -> None:
        """Cold start (no production) should always promote."""
        result = registry_no_prod.compare_to_production(0.4)

        assert result["should_promote"] is True
        assert "cold start" in result["reason"]
        assert result["production_version"] is None
        assert result["production_rmse"] is None
        assert result["candidate_rmse"] == 0.4
        assert result["delta_pct"] is None

    def test_cold_start_with_bad_candidate(self, registry_no_prod: ModelRegistry) -> None:
        """Even a bad candidate should promote on cold start."""
        result = registry_no_prod.compare_to_production(10.0)

        assert result["should_promote"] is True
        assert "cold start" in result["reason"]


class TestCompareToProductionWithExisting:
    """Tests for comparison against existing production model."""

    def test_candidate_better_than_prod(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate with better RMSE should promote."""
        # Production has val_rmse=0.5, candidate is 0.4 (better)
        result = registry_with_prod.compare_to_production(0.4)

        assert result["should_promote"] is True
        assert result["production_version"] == "20260315"
        assert result["production_rmse"] == 0.5
        assert result["candidate_rmse"] == 0.4
        assert result["delta_pct"] < 0  # Negative = candidate is better
        assert result["delta_pct"] == pytest.approx(-20.0, rel=0.01)

    def test_candidate_worse_than_prod(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate with worse RMSE beyond tolerance should NOT promote."""
        # Production has val_rmse=0.5, candidate is 0.6 (20% worse)
        result = registry_with_prod.compare_to_production(0.6, tolerance=0.02)

        assert result["should_promote"] is False
        assert "worse than" in result["reason"]
        assert result["delta_pct"] > 0  # Positive = candidate is worse
        assert result["delta_pct"] == pytest.approx(20.0, rel=0.01)

    def test_candidate_within_tolerance(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate slightly worse but within tolerance should promote."""
        # Production has val_rmse=0.5, candidate is 0.505 (1% worse)
        # Default tolerance is 2%
        result = registry_with_prod.compare_to_production(0.505, tolerance=0.02)

        assert result["should_promote"] is True
        assert "within" in result["reason"]
        assert result["delta_pct"] == pytest.approx(1.0, rel=0.01)

    def test_candidate_at_tolerance_boundary(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate exactly at tolerance boundary should promote."""
        # Production has val_rmse=0.5, candidate is 0.51 (2% worse, at boundary)
        result = registry_with_prod.compare_to_production(0.51, tolerance=0.02)

        assert result["should_promote"] is True

    def test_candidate_just_beyond_tolerance(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate just beyond tolerance should NOT promote."""
        # Production has val_rmse=0.5, candidate is 0.511 (2.2% worse)
        result = registry_with_prod.compare_to_production(0.511, tolerance=0.02)

        assert result["should_promote"] is False

    def test_strict_tolerance(self, registry_with_prod: ModelRegistry) -> None:
        """With tolerance=0, only strictly better candidates promote."""
        # Production has val_rmse=0.5, candidate is 0.501 (slightly worse)
        result = registry_with_prod.compare_to_production(0.501, tolerance=0.0)

        assert result["should_promote"] is False

    def test_exact_match_promotes(self, registry_with_prod: ModelRegistry) -> None:
        """Candidate with exact same RMSE should promote."""
        result = registry_with_prod.compare_to_production(0.5, tolerance=0.0)

        assert result["should_promote"] is True
        assert result["delta_pct"] == 0.0


class TestCompareToProductionEdgeCases:
    """Edge cases and error handling."""

    def test_prod_has_no_rmse_in_metadata(self, registry_with_prod_no_rmse: ModelRegistry) -> None:
        """If production has no val_rmse, candidate should promote."""
        result = registry_with_prod_no_rmse.compare_to_production(0.4)

        assert result["should_promote"] is True
        assert "no val_rmse" in result["reason"]

    def test_custom_tolerance_value(self, registry_with_prod: ModelRegistry) -> None:
        """Test with custom tolerance value."""
        # Production has val_rmse=0.5, candidate is 0.55 (10% worse)
        # With 10% tolerance, should still promote
        result = registry_with_prod.compare_to_production(0.55, tolerance=0.10)

        assert result["should_promote"] is True

        # With 5% tolerance, should NOT promote
        result = registry_with_prod.compare_to_production(0.55, tolerance=0.05)

        assert result["should_promote"] is False

    def test_result_structure(self, registry_with_prod: ModelRegistry) -> None:
        """Verify all expected keys are present in result."""
        result = registry_with_prod.compare_to_production(0.45)

        expected_keys = {
            "should_promote",
            "reason",
            "production_version",
            "production_rmse",
            "candidate_rmse",
            "delta_pct",
        }
        assert set(result.keys()) == expected_keys
