"""
Tests for Boston Police Stations Data Pipeline
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from bias_detection_stations import StationsBiasDetector
from ingest_stations import validate_stations_data
from preprocess_stations import generate_statistics, haversine_distance, preprocess_stations_data

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_data(tmp_path):
    """Create a sample raw police stations CSV for testing"""
    data = {
        "OBJECTID": [1, 2, 3, 4],
        "NAME": [
            "District A-1 Police Station",
            "District B-2 Police Station",
            "Boston Police Headquarters",
            "District C-6 Police Station",
        ],
        "ADDRESS": ["40 Sudbury St", "2400 Washington St", "1 Schroeder Plz", "101 W Broadway"],
        "NEIGHBORHOOD": ["Boston", "Roxbury", "Roxbury", "South Boston"],
        "CITY": ["Boston", "Boston", "Boston", "Boston"],
        "ZIP": ["02114", "02119", "02120", "02127"],
        "FT_SQFT": [6882, 10809, 194000, 8528],
        "STORY_HT": [5, 3, 4, 2],
        "POINT_X": [-71.060306, -71.085683, -71.090746, -71.054935],
        "POINT_Y": [42.361816, 42.328376, 42.334183, 42.341155],
        "District": ["A1", "B2", None, "C6"],
        "latitude": [42.361816, 42.328376, 42.334183, 42.341155],
        "longitude": [-71.060306, -71.085683, -71.090746, -71.054935],
    }
    df = pd.DataFrame(data)
    raw_path = tmp_path / "police_stations.csv"
    df.to_csv(raw_path, index=False)
    return str(raw_path)


@pytest.fixture
def sample_processed_data(tmp_path):
    """Create a sample processed police stations CSV for testing"""
    data = {
        "NAME": [
            "DISTRICT A-1 POLICE STATION",
            "DISTRICT B-2 POLICE STATION",
            "BOSTON POLICE HEADQUARTERS",
            "DISTRICT C-6 POLICE STATION",
        ],
        "NEIGHBORHOOD": ["BOSTON", "ROXBURY", "ROXBURY", "SOUTH BOSTON"],
        "DISTRICT": ["A1", "B2", "NONE", "C6"],
        "LAT": [42.361816, 42.328376, 42.334183, 42.341155],
        "LON": [-71.060306, -71.085683, -71.090746, -71.054935],
        "FT_SQFT": [6882, 10809, 194000, 8528],
        "STORY_HT": [5, 3, 4, 2],
        "VALID_COORDS": [1, 1, 1, 1],
        "DIST_FROM_CENTER_KM": [0.5, 3.2, 2.8, 2.4],
        "SIZE_CATEGORY": ["Small", "Medium", "Large", "Small"],
        "IS_HEADQUARTERS": [0, 0, 1, 0],
        "ZONE": ["Inner", "Inner", "Inner", "Inner"],
        "FT_SQFT_NORMALIZED": [-0.5, -0.2, 2.1, -0.4],
        "FT_SQFT_MISSING": [0, 0, 0, 0],
    }
    df = pd.DataFrame(data)
    processed_path = tmp_path / "police_stations_clean.csv"
    df.to_csv(processed_path, index=False)
    return str(processed_path)


# ─── Ingestion Tests ──────────────────────────────────────────────────────────


class TestValidateStationsData:
    def test_valid_data_passes(self, sample_raw_data):
        result = validate_stations_data(sample_raw_data)
        assert result["total_rows"] == 4
        assert result["critical_issues"] == []

    def test_empty_file_returns_critical_issue(self, tmp_path):
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        result = validate_stations_data(str(empty_path))
        assert "Dataset is empty" in result["critical_issues"]

    def test_missing_expected_columns(self, tmp_path):
        df = pd.DataFrame({"col1": [1, 2]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        result = validate_stations_data(str(path))
        assert any("Missing expected columns" in issue for issue in result["critical_issues"])

    def test_duplicate_detection(self, sample_raw_data):
        result = validate_stations_data(sample_raw_data)
        assert "duplicate_rows" in result
        assert result["duplicate_rows"] == 0

    def test_missing_values_counted(self, sample_raw_data):
        result = validate_stations_data(sample_raw_data)
        assert "missing_values" in result
        assert result["missing_values"] >= 0

    def test_returns_column_list(self, sample_raw_data):
        result = validate_stations_data(sample_raw_data)
        assert isinstance(result["columns"], list)
        assert len(result["columns"]) > 0


# ─── Preprocessing Tests ──────────────────────────────────────────────────────


class TestHaversineDistance:
    def test_same_point_is_zero(self):
        dist = haversine_distance(42.36, -71.06, 42.36, -71.06)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # Boston City Hall to Fenway Park ~3.7 km
        dist = haversine_distance(42.3601, -71.0589, 42.3467, -71.0972)
        assert 3.0 < dist < 4.5

    def test_returns_float(self):
        dist = haversine_distance(42.36, -71.06, 42.34, -71.09)
        assert isinstance(dist, float)


class TestPreprocessStationsData:
    def test_output_file_created(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        result = preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        assert Path(result).exists()

    def test_feature_columns_added(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        expected_features = [
            "LAT",
            "LON",
            "VALID_COORDS",
            "DIST_FROM_CENTER_KM",
            "IS_HEADQUARTERS",
            "ZONE",
            "FT_SQFT_NORMALIZED",
        ]
        for col in expected_features:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_headquarters_flagged_correctly(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert "IS_HEADQUARTERS" in df.columns
        assert df["IS_HEADQUARTERS"].sum() == 1  # Only HQ should be flagged

    def test_valid_coords_flagged(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert df["VALID_COORDS"].isin([0, 1]).all()

    def test_no_duplicate_rows(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert df.duplicated().sum() == 0

    def test_text_fields_uppercased(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        if "NAME" in df.columns:
            valid_names = df["NAME"].dropna()
            assert valid_names.str.isupper().all()

    def test_dist_from_center_positive(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_stations_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert (df["DIST_FROM_CENTER_KM"].dropna() >= 0).all()


class TestGenerateStatistics:
    def test_returns_expected_keys(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        expected_keys = [
            "total_stations",
            "columns",
            "zone_distribution",
            "size_distribution",
            "avg_dist_from_center_km",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_total_stations_correct(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        assert stats["total_stations"] == 4

    def test_headquarters_count(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        assert stats["headquarters_count"] == 1


# ─── Bias Detection Tests ─────────────────────────────────────────────────────


class TestStationsBiasDetector:
    def test_initialization(self, sample_processed_data):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        assert len(detector.df) == 4

    def test_geographic_bias_runs(self, sample_processed_data):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        result = detector.detect_geographic_coverage_bias()
        assert result is not None

    def test_size_equity_bias_runs(self, sample_processed_data):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        result = detector.detect_size_equity_bias()
        assert result is not None
        assert "overall" in result

    def test_zone_distribution_bias_runs(self, sample_processed_data):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        result = detector.detect_zone_distribution_bias()
        assert result is not None

    def test_report_generated(self, sample_processed_data, tmp_path):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "test_bias_report.txt")
        report = detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).exists()
        assert "geographic_coverage_bias" in report
        assert "size_equity_bias" in report

    def test_report_file_not_empty(self, sample_processed_data, tmp_path):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "test_bias_report.txt")
        detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).stat().st_size > 0

    def test_bias_report_has_recommendations_key(self, sample_processed_data):
        detector = StationsBiasDetector(data_path=sample_processed_data)
        detector.detect_geographic_coverage_bias()
        assert "mitigation_recommendations" in detector.bias_report
