"""
Tests for Boston Fire Incidents Data Pipeline
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from bias_detection_fire import FireBiasDetector
from ingest_fire import validate_fire_data
from preprocess_fire import categorize_incident_severity, generate_statistics, preprocess_fire_data

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_data(tmp_path):
    data = {
        "Incident Number": ["13-0000001", "13-0000002", "13-0000003", "13-0000004", "13-0000005"],
        "Exposure Number": [0, 0, 0, 0, 0],
        "Alarm Date": ["01/01/13", "01/01/13", "01/02/13", "01/02/13", "01/03/13"],
        "Alarm Time": ["0:04:38", "1:15:00", "14:30:00", "22:00:00", "8:45:00"],
        "Incident Type": [100, 700, 321, 743, 111],
        "Incident Description": ["Fire", "False Alarm", "EMS", "Smoke Detector", "Building Fire"],
        "Estimated Property Loss": [5000, 0, 0, 0, 25000],
        "Estimated Content Loss": [1000, 0, 0, 0, 5000],
        "District": ["7", "9", "7", "3", "9"],
        "Neighborhood": ["Roxbury", "Hyde Park", "Roxbury", "Dorchester", "Hyde Park"],
        "Zip": ["02119", "02136", "02119", "02122", "02136"],
        "Property Use": ["429", "500", "429", "429", "400"],
        "Property Description": [
            "Multifamily",
            "Mercantile",
            "Multifamily",
            "Multifamily",
            "Residential",
        ],
        "Street Number": [100, 200, 300, 400, 500],
        "Street Name": ["Main St", "Oak Ave", "Elm St", "Pine Rd", "Maple Dr"],
    }
    df = pd.DataFrame(data)
    raw_path = tmp_path / "fire_incidents.csv"
    df.to_csv(raw_path, index=False)
    return str(raw_path)


@pytest.fixture
def sample_processed_data(tmp_path):
    data = {
        "Incident Number": ["13-0000001", "13-0000002", "13-0000003", "13-0000004", "13-0000005"],
        "Alarm Date": ["01/01/13", "01/01/13", "01/02/13", "01/02/13", "01/03/13"],
        "Alarm Time": ["0:04:38", "1:15:00", "14:30:00", "22:00:00", "8:45:00"],
        "Incident Type": [100, 700, 321, 743, 111],
        "Estimated Property Loss": [5000.0, 0.0, 0.0, 0.0, 25000.0],
        "Estimated Content Loss": [1000.0, 0.0, 0.0, 0.0, 5000.0],
        "District": ["7", "9", "7", "3", "9"],
        "Neighborhood": ["ROXBURY", "HYDE PARK", "ROXBURY", "DORCHESTER", "HYDE PARK"],
        "alarm_datetime": [
            "2013-01-01 00:04:38",
            "2013-01-01 01:15:00",
            "2013-01-02 14:30:00",
            "2013-01-02 22:00:00",
            "2013-01-03 08:45:00",
        ],
        "year": [2013, 2013, 2013, 2013, 2013],
        "month": [1, 1, 1, 1, 1],
        "day_of_week": [1, 1, 2, 2, 3],
        "hour": [0, 1, 14, 22, 8],
        "is_weekend": [0, 0, 0, 0, 0],
        "time_of_day": ["Night", "Night", "Afternoon", "Evening", "Morning"],
        "severity_category": ["Fire", "False Alarm", "Rescue", "False Alarm", "Fire"],
        "total_loss": [6000.0, 0.0, 0.0, 0.0, 30000.0],
        "has_loss": [1, 0, 0, 0, 1],
        "loss_category": ["Minor", "No Loss", "No Loss", "No Loss", "Moderate"],
        "district_incident_count": [2, 2, 2, 1, 2],
    }
    df = pd.DataFrame(data)
    processed_path = tmp_path / "fire_incidents_clean.csv"
    df.to_csv(processed_path, index=False)
    return str(processed_path)


# ─── Ingestion Tests ──────────────────────────────────────────────────────────


class TestValidateFireData:
    def test_valid_data_passes(self, sample_raw_data):
        result = validate_fire_data(sample_raw_data)
        assert result["total_rows"] == 5
        assert result["critical_issues"] == []

    def test_empty_file_returns_critical_issue(self, tmp_path):
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        result = validate_fire_data(str(empty_path))
        assert "Dataset is empty" in result["critical_issues"]

    def test_missing_expected_columns(self, tmp_path):
        df = pd.DataFrame({"col1": [1, 2]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        result = validate_fire_data(str(path))
        assert any("Missing expected columns" in issue for issue in result["critical_issues"])

    def test_duplicate_detection(self, sample_raw_data):
        result = validate_fire_data(sample_raw_data)
        assert result["duplicate_rows"] == 0

    def test_column_count(self, sample_raw_data):
        result = validate_fire_data(sample_raw_data)
        assert result["column_count"] > 0

    def test_returns_column_list(self, sample_raw_data):
        result = validate_fire_data(sample_raw_data)
        assert isinstance(result["columns"], list)


# ─── Preprocessing Tests ──────────────────────────────────────────────────────


class TestCategorizeIncidentSeverity:
    def test_fire_category(self):
        assert categorize_incident_severity(100) == "Fire"
        assert categorize_incident_severity(111) == "Fire"

    def test_false_alarm_category(self):
        assert categorize_incident_severity(700) == "False Alarm"
        assert categorize_incident_severity(743) == "False Alarm"

    def test_rescue_category(self):
        assert categorize_incident_severity(321) == "Rescue"

    def test_hazmat_category(self):
        assert categorize_incident_severity(400) == "Hazmat"

    def test_nan_returns_unknown(self):
        assert categorize_incident_severity(None) == "Unknown"

    def test_explosion_category(self):
        assert categorize_incident_severity(200) == "Explosion"


class TestPreprocessFireData:
    def test_output_file_created(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        result = preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        assert Path(result).exists()

    def test_feature_columns_added(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        expected = [
            "severity_category",
            "total_loss",
            "has_loss",
            "time_of_day",
            "is_weekend",
            "district_incident_count",
        ]
        for col in expected:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_duplicates(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert df.duplicated().sum() == 0

    def test_total_loss_is_sum(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        if "total_loss" in df.columns:
            expected = df["Estimated Property Loss"] + df["Estimated Content Loss"]
            assert (df["total_loss"] == expected).all()

    def test_has_loss_binary(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        assert df["has_loss"].isin([0, 1]).all()

    def test_text_fields_uppercased(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        if "Neighborhood" in df.columns:
            valid = df["Neighborhood"].dropna()
            assert valid.str.isupper().all()

    def test_temporal_features_extracted(self, sample_raw_data, tmp_path):
        output_path = str(tmp_path / "processed.csv")
        preprocess_fire_data(input_path=sample_raw_data, output_path=output_path)
        df = pd.read_csv(output_path)
        for col in ["year", "month", "hour", "is_weekend"]:
            assert col in df.columns


class TestGenerateFireStatistics:
    def test_returns_expected_keys(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        for key in [
            "total_records",
            "severity_distribution",
            "district_distribution",
            "total_property_loss",
        ]:
            assert key in stats

    def test_total_records_correct(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        assert stats["total_records"] == 5

    def test_incidents_with_loss(self, sample_processed_data):
        stats = generate_statistics(sample_processed_data)
        assert stats["incidents_with_loss"] == 2


# ─── Bias Detection Tests ─────────────────────────────────────────────────────


class TestFireBiasDetector:
    def test_initialization(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        assert len(detector.df) == 5

    def test_geographic_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        result = detector.detect_geographic_bias()
        assert result is not None

    def test_temporal_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        result = detector.detect_temporal_bias()
        assert result is not None

    def test_severity_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        result = detector.detect_severity_bias()
        assert result is not None
        assert "overall_distribution" in result

    def test_loss_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        result = detector.detect_loss_bias()
        assert result is not None

    def test_report_generated(self, sample_processed_data, tmp_path):
        detector = FireBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "fire_bias_report.txt")
        report = detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).exists()
        assert "geographic_bias" in report
        assert "temporal_bias" in report
        assert "severity_bias" in report
        assert "loss_bias" in report

    def test_report_not_empty(self, sample_processed_data, tmp_path):
        detector = FireBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "fire_bias_report.txt")
        detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).stat().st_size > 0

    def test_bias_report_has_recommendations(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_geographic_bias()
        assert "mitigation_recommendations" in detector.bias_report
