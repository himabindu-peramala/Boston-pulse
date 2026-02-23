"""
Unit tests for CrimePreprocessor.

Tests the crime data preprocessing and validation.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.datasets.crime.preprocess import CrimePreprocessor, preprocess_crime_data


class TestCrimePreprocessor:
    """Test cases for CrimePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a CrimePreprocessor instance."""
        return CrimePreprocessor()

    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data matching API format."""
        return pd.DataFrame(
            {
                "INCIDENT_NUMBER": ["I2024001", "I2024002", "I2024003"],
                "OFFENSE_CODE": [3115, 619, 801],
                "OFFENSE_CODE_GROUP": ["Investigate Property", "Larceny", "Assault"],
                "OFFENSE_DESCRIPTION": [
                    "INVESTIGATE PROPERTY",
                    "LARCENY THEFT FROM MV",
                    "SIMPLE ASSAULT",
                ],
                "DISTRICT": ["A1", "B2", "C6"],
                "REPORTING_AREA": ["101", "202", "303"],
                "SHOOTING": ["N", "N", "Y"],
                "OCCURRED_ON_DATE": [
                    "2024-01-15T14:30:00+00:00",
                    "2024-01-15T16:45:00+00:00",
                    "2024-01-15T22:00:00+00:00",
                ],
                "YEAR": [2024, 2024, 2024],
                "MONTH": [1, 1, 1],
                "DAY_OF_WEEK": ["Monday", "Monday", "Monday"],
                "HOUR": [14, 16, 22],
                "UCR_PART": ["Part Three", "Part One", "Part One"],
                "STREET": ["MAIN ST", "WASHINGTON ST", "BROADWAY"],
                "Lat": [42.3601, 42.3489, 42.3560],
                "Long": [-71.0589, -71.0765, -71.0520],
            }
        )

    def test_get_dataset_name(self, preprocessor):
        """Test dataset name is correct."""
        assert preprocessor.get_dataset_name() == "crime"

    def test_get_required_columns(self, preprocessor):
        """Test required columns are defined."""
        required = preprocessor.get_required_columns()
        assert "incident_number" in required
        assert "offense_category" in required
        assert "occurred_on_date" in required
        assert "district" in required

    def test_get_column_mappings(self, preprocessor):
        """Test column mappings are defined."""
        mappings = preprocessor.get_column_mappings()
        assert "INCIDENT_NUMBER" in mappings
        assert mappings["INCIDENT_NUMBER"] == "incident_number"
        assert mappings["OFFENSE_CODE_GROUP"] == "offense_category"

    def test_run_success(self, preprocessor, sample_raw_data):
        """Test successful preprocessing run."""
        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        print(result)
        assert result.success
        assert result.dataset == "crime"
        assert result.rows_input == 3
        assert result.rows_output > 0

    def test_run_column_renaming(self, preprocessor, sample_raw_data):
        """Test that columns are renamed correctly."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert "incident_number" in df.columns
        assert "offense_category" in df.columns
        assert "occurred_on_date" in df.columns
        assert "INCIDENT_NUMBER" not in df.columns

    def test_run_shooting_conversion(self, preprocessor, sample_raw_data):
        """Test shooting field conversion to boolean."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert "shooting" in df.columns
        assert df["shooting"].dtype == bool
        assert df["shooting"].iloc[2]

    def test_run_datetime_parsing(self, preprocessor, sample_raw_data):
        """Test datetime field parsing."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert pd.api.types.is_datetime64_any_dtype(df["occurred_on_date"])

    def test_run_coordinate_validation(self, preprocessor, sample_raw_data):
        """Test coordinate validation."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert df["lat"].between(42.2, 42.4).all()
        assert df["long"].between(-71.2, -70.9).all()

    def test_run_district_standardization(self, preprocessor, sample_raw_data):
        """Test district standardization."""
        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        assert df["district"].str.isupper().all()

    def test_run_drops_duplicates(self, preprocessor, sample_raw_data):
        """Test duplicate removal."""
        # Add a duplicate
        df_with_dup = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]])

        result = preprocessor.run(df_with_dup, execution_date="2024-01-15")

        # Should have dropped 1 duplicate
        assert result.rows_dropped >= 1

    def test_run_handles_missing_coords(self, preprocessor, sample_raw_data):
        """Test handling of missing coordinates."""
        sample_raw_data.loc[0, "Lat"] = np.nan
        sample_raw_data.loc[0, "Long"] = np.nan

        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        # Record should still exist but with NaN coords
        assert len(df) > 0

    def test_run_filters_invalid_coords(self, preprocessor, sample_raw_data):
        """Test filtering of invalid coordinates."""
        sample_raw_data.loc[0, "Lat"] = 0  # Outside Boston
        sample_raw_data.loc[0, "Long"] = 0

        preprocessor.run(sample_raw_data, execution_date="2024-01-15")
        df = preprocessor.get_data()

        # Coords should be set to NaN but record kept
        assert pd.isna(df.loc[df["incident_number"] == "I2024001", "lat"].iloc[0])

    def test_run_filters_invalid_dates(self, preprocessor, sample_raw_data):
        """Test filtering of invalid dates."""
        sample_raw_data.loc[0, "OCCURRED_ON_DATE"] = "invalid-date"

        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        # Should have dropped the invalid date record
        assert result.rows_dropped >= 1

    def test_run_filters_future_dates(self, preprocessor, sample_raw_data):
        """Test filtering of future dates."""
        future_date = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        sample_raw_data.loc[0, "OCCURRED_ON_DATE"] = future_date

        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        # Should have dropped the future date record
        assert result.rows_dropped >= 1

    def test_run_transformations_logged(self, preprocessor, sample_raw_data):
        """Test that transformations are logged."""
        result = preprocessor.run(sample_raw_data, execution_date="2024-01-15")

        assert len(result.transformations_applied) > 0

    def test_run_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        result = preprocessor.run(df, execution_date="2024-01-15")

        # Should fail gracefully
        assert not result.success or result.rows_output == 0

    def test_run_missing_columns(self, preprocessor):
        """Test handling of missing required columns."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})

        result = preprocessor.run(df, execution_date="2024-01-15")

        assert not result.success


class TestPreprocessingMethods:
    """Test individual preprocessing methods."""

    @pytest.fixture
    def preprocessor(self):
        """Create a CrimePreprocessor instance."""
        return CrimePreprocessor()

    def test_process_shooting_field_y(self, preprocessor):
        """Test shooting field conversion with 'Y'."""
        df = pd.DataFrame({"shooting": ["Y", "N", "y", "n"]})
        df = preprocessor._process_shooting_field(df)

        assert df["shooting"].tolist() == [True, False, True, False]

    def test_process_shooting_field_numeric(self, preprocessor):
        """Test shooting field conversion with numeric values."""
        df = pd.DataFrame({"shooting": [1, 0, "1", "0"]})
        df = preprocessor._process_shooting_field(df)

        assert df["shooting"].tolist() == [True, False, True, False]

    def test_standardize_coordinates(self, preprocessor):
        """Test coordinate standardization."""
        df = pd.DataFrame(
            {
                "lat": ["42.35", "42.36", "0"],
                "long": ["-71.05", "-71.06", "0"],
            }
        )
        df = preprocessor.standardize_coordinates(df)

        # Invalid coords (0, 0) should be filtered
        assert len(df) == 2
        assert df["lat"].dtype == float

    def test_standardize_datetime(self, preprocessor):
        """Test datetime standardization."""
        df = pd.DataFrame({"date": ["2024-01-15T14:30:00", "2024-01-16T10:00:00"]})
        df = preprocessor.standardize_datetime(df, "date")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert "date_year" in df.columns
        assert "date_month" in df.columns
        assert "date_hour" in df.columns


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_preprocess_crime_data(self):
        """Test preprocess_crime_data convenience function."""
        df = pd.DataFrame(
            {
                "INCIDENT_NUMBER": ["I001"],
                "OFFENSE_CODE": [100],
                "OFFENSE_CODE_GROUP": ["Test"],
                "DISTRICT": ["A1"],
                "SHOOTING": ["N"],
                "OCCURRED_ON_DATE": ["2024-01-15T14:30:00"],
                "YEAR": [2024],
                "MONTH": [1],
                "HOUR": [14],
                "Lat": [42.36],
                "Long": [-71.05],
            }
        )

        result = preprocess_crime_data(df, execution_date="2024-01-15")

        assert isinstance(result, dict)
        assert "success" in result or "rows_output" in result
