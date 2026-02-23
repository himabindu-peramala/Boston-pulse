"""
Unit tests for crime data preprocessing
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.preprocess_crime import generate_statistics, preprocess_crime_data


class TestPreprocessing:
    """Test suite for preprocessing functions"""

    def test_temporal_features_extracted(self, tmp_path):
        """Test that temporal features are correctly extracted"""
        # Create test data
        test_data = {
            "INCIDENT_NUMBER": ["I12345", "I12346"],
            "OFFENSE_CODE": [619, 724],
            "DISTRICT": ["D14", "C11"],
            "OCCURRED_ON_DATE": ["2023-06-15 14:30:00", "2023-12-25 22:45:00"],
            "Lat": [42.3601, 42.3602],
            "Long": [-71.0589, -71.0590],
        }
        df = pd.DataFrame(test_data)

        # Save to temp file
        input_file = tmp_path / "test_raw.csv"
        output_file = tmp_path / "test_processed.csv"
        df.to_csv(input_file, index=False)

        # Run preprocessing
        preprocess_crime_data(str(input_file), str(output_file))

        # Load processed data
        result_df = pd.read_csv(output_file)

        # Check temporal features exist
        assert "year" in result_df.columns
        assert "month" in result_df.columns
        assert "hour" in result_df.columns
        assert "day_of_week" in result_df.columns
        assert "time_of_day" in result_df.columns

        # Check values are correct
        assert result_df["year"].iloc[0] == 2023
        assert result_df["month"].iloc[0] == 6
        assert result_df["hour"].iloc[0] == 14

    def test_duplicates_removed(self, tmp_path):
        """Test that duplicate rows are removed"""
        # Create data with duplicates
        test_data = {
            "INCIDENT_NUMBER": ["I12345", "I12345", "I12346"],
            "OFFENSE_CODE": [619, 619, 724],
            "DISTRICT": ["D14", "D14", "C11"],
            "OCCURRED_ON_DATE": ["2023-01-01", "2023-01-01", "2023-01-02"],
        }
        df = pd.DataFrame(test_data)

        input_file = tmp_path / "test_raw_dupes.csv"
        output_file = tmp_path / "test_processed_dupes.csv"
        df.to_csv(input_file, index=False)

        # Run preprocessing
        preprocess_crime_data(str(input_file), str(output_file))

        # Load result
        result_df = pd.read_csv(output_file)

        # Should have removed duplicates
        assert len(result_df) < len(df)

    def test_severity_categorization(self, tmp_path):
        """Test that crimes are categorized by severity"""
        test_data = {
            "INCIDENT_NUMBER": ["I1", "I2", "I3"],
            "OFFENSE_CODE": [100, 1500, 3500],  # Serious, Moderate, Minor
            "DISTRICT": ["D14", "C11", "A1"],
            "OCCURRED_ON_DATE": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }
        df = pd.DataFrame(test_data)

        input_file = tmp_path / "test_raw_severity.csv"
        output_file = tmp_path / "test_processed_severity.csv"
        df.to_csv(input_file, index=False)

        # Run preprocessing
        preprocess_crime_data(str(input_file), str(output_file))

        # Load result
        result_df = pd.read_csv(output_file)

        # Check severity column exists
        assert "severity" in result_df.columns

        # Check categorization
        assert result_df["severity"].iloc[0] == "Serious"
        assert result_df["severity"].iloc[1] == "Moderate"
        assert result_df["severity"].iloc[2] == "Minor"

    def test_text_standardization(self, tmp_path):
        """Test that text fields are standardized"""
        test_data = {
            "INCIDENT_NUMBER": ["I12345"],
            "OFFENSE_CODE": [619],
            "OFFENSE_DESCRIPTION": ["  larceny all others  "],  # Lowercase with spaces
            "DISTRICT": ["d14"],  # Lowercase
            "OCCURRED_ON_DATE": ["2023-01-01"],
        }
        df = pd.DataFrame(test_data)

        input_file = tmp_path / "test_raw_text.csv"
        output_file = tmp_path / "test_processed_text.csv"
        df.to_csv(input_file, index=False)

        # Run preprocessing
        preprocess_crime_data(str(input_file), str(output_file))

        # Load result
        result_df = pd.read_csv(output_file)

        # Check text is uppercase and trimmed
        assert result_df["OFFENSE_DESCRIPTION"].iloc[0] == "LARCENY ALL OTHERS"
        assert result_df["DISTRICT"].iloc[0] == "D14"


class TestStatisticsGeneration:
    """Test suite for statistics generation"""

    def test_statistics_generation(self, tmp_path):
        """Test that statistics are generated correctly"""
        # Create test processed data
        test_data = {
            "INCIDENT_NUMBER": ["I1", "I2", "I3"],
            "OFFENSE_DESCRIPTION": ["LARCENY", "LARCENY", "ASSAULT"],
            "DISTRICT": ["D14", "D14", "C11"],
            "OCCURRED_ON_DATE": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "severity": ["Minor", "Minor", "Serious"],
            "time_of_day": ["Morning", "Afternoon", "Night"],
        }
        df = pd.DataFrame(test_data)

        test_file = tmp_path / "test_processed_stats.csv"
        df.to_csv(test_file, index=False)

        # Generate statistics
        stats = generate_statistics(str(test_file))

        # Check statistics structure
        assert "total_records" in stats
        assert "top_offenses" in stats
        assert "district_distribution" in stats
        assert "severity_distribution" in stats

        # Check values
        assert stats["total_records"] == 3
        assert stats["top_offenses"]["LARCENY"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
