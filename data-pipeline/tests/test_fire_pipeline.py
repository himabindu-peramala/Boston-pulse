"""
Tests for Boston Fire Incidents Data Pipeline
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.fire.bias_detection import FireBiasDetector
from src.datasets.fire.ingester import FireIngester
from src.datasets.fire.preprocessor import FirePreprocessor


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_data(tmp_path):
    data = {
        "Incident Number": [
            "13-0000001",
            "13-0000002",
            "13-0000003",
            "13-0000004",
            "13-0000005",
        ],
        "Exposure Number": [0, 0, 0, 0, 0],
        "Alarm Date": ["01/01/13", "01/01/13", "01/02/13", "01/02/13", "01/03/13"],
        "Alarm Time": ["0:04:38", "1:15:00", "14:30:00", "22:00:00", "8:45:00"],
        "Incident Type": [100, 700, 321, 743, 111],
        "Incident Description": [
            "Fire",
            "False Alarm",
            "EMS",
            "Smoke Detector",
            "Building Fire",
        ],
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
        "Incident Number": [
            "13-0000001",
            "13-0000002",
            "13-0000003",
            "13-0000004",
            "13-0000005",
        ],
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


# ─── Ingester Tests ───────────────────────────────────────────────────────────


class TestFireIngester:
    def test_get_dataset_name(self):
        ingester = FireIngester()
        assert ingester.get_dataset_name() == "fire"

    def test_get_watermark_field(self):
        ingester = FireIngester()
        assert ingester.get_watermark_field() is not None

    def test_get_primary_key(self):
        ingester = FireIngester()
        assert ingester.get_primary_key() is not None


# ─── Preprocessor Tests ───────────────────────────────────────────────────────


class TestFirePreprocessor:
    def test_get_dataset_name(self):
        preprocessor = FirePreprocessor()
        assert preprocessor.get_dataset_name() == "fire"

    def test_get_required_columns(self):
        preprocessor = FirePreprocessor()
        cols = preprocessor.get_required_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_column_mappings(self):
        preprocessor = FirePreprocessor()
        mappings = preprocessor.get_column_mappings()
        assert isinstance(mappings, dict)

    def test_get_dtype_mappings(self):
        preprocessor = FirePreprocessor()
        mappings = preprocessor.get_dtype_mappings()
        assert isinstance(mappings, dict)

    def test_transform_basic(self, sample_raw_data):
        preprocessor = FirePreprocessor()
        df = pd.read_csv(sample_raw_data)
        result = preprocessor.transform(df)
        assert result is not None
        assert len(result) > 0


# ─── Bias Detection Tests ─────────────────────────────────────────────────────


class TestFireBiasDetector:
    def test_initialization(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        assert len(detector.df) == 5

    def test_geographic_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_geographic_bias()
        assert "geographic_bias" in detector.bias_report

    def test_temporal_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_temporal_bias()
        assert "temporal_bias" in detector.bias_report

    def test_severity_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_severity_bias()
        assert "severity_bias" in detector.bias_report

    def test_loss_bias_runs(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_loss_bias()
        assert "loss_bias" in detector.bias_report

    def test_report_generated(self, sample_processed_data, tmp_path):
        detector = FireBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "fire_bias_report.txt")
        report = detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).exists()
        assert "geographic_bias" in report
        assert "temporal_bias" in report

    def test_report_not_empty(self, sample_processed_data, tmp_path):
        detector = FireBiasDetector(data_path=sample_processed_data)
        report_path = str(tmp_path / "fire_bias_report.txt")
        detector.generate_bias_report(output_path=report_path)
        assert Path(report_path).stat().st_size > 0

    def test_bias_report_has_recommendations(self, sample_processed_data):
        detector = FireBiasDetector(data_path=sample_processed_data)
        detector.detect_geographic_bias()
        assert "mitigation_recommendations" in detector.bias_report
