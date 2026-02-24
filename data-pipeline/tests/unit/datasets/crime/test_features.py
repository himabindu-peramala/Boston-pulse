"""
Unit tests for CrimeFeatureBuilder.

Tests the crime feature building and aggregation.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.datasets.crime.features import CrimeFeatureBuilder, build_crime_features


class TestCrimeFeatureBuilder:
    """Test cases for CrimeFeatureBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a CrimeFeatureBuilder instance."""
        return CrimeFeatureBuilder()

    @pytest.fixture
    def sample_processed_data(self):
        """Sample processed data for feature building."""
        base_date = datetime(2024, 1, 15)
        return pd.DataFrame(
            {
                "incident_number": [f"I{i}" for i in range(10)],
                "offense_code": [100, 200, 300, 100, 200, 300, 100, 200, 300, 100],
                "offense_category": [
                    "Assault",
                    "Larceny",
                    "Vandalism",
                    "Assault",
                    "Larceny",
                    "Vandalism",
                    "Robbery",
                    "Larceny",
                    "Vandalism",
                    "Assault",
                ],
                "district": [
                    "A1",
                    "A1",
                    "A1",
                    "B2",
                    "B2",
                    "B2",
                    "C6",
                    "C6",
                    "C6",
                    "A1",
                ],
                "shooting": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                "occurred_on_date": pd.to_datetime(
                    [base_date - timedelta(days=i) for i in range(10)]
                ),
                "year": [2024] * 10,
                "month": [1] * 10,
                "day_of_week": [
                    "Monday",
                    "Sunday",
                    "Saturday",
                    "Friday",
                    "Thursday",
                    "Wednesday",
                    "Tuesday",
                    "Monday",
                    "Sunday",
                    "Saturday",
                ],
                "hour": [14, 22, 3, 10, 15, 20, 2, 12, 18, 23],
                "lat": [
                    42.360,
                    42.360,
                    42.361,
                    42.350,
                    42.350,
                    42.351,
                    42.340,
                    42.340,
                    42.341,
                    42.360,
                ],
                "long": [
                    -71.058,
                    -71.058,
                    -71.059,
                    -71.070,
                    -71.070,
                    -71.071,
                    -71.080,
                    -71.080,
                    -71.081,
                    -71.058,
                ],
            }
        )

    def test_get_dataset_name(self, builder):
        """Test dataset name is correct."""
        assert builder.get_dataset_name() == "crime"

    def test_get_entity_key(self, builder):
        """Test entity key is correct."""
        assert builder.get_entity_key() == "grid_cell"

    def test_get_feature_definitions(self, builder):
        """Test feature definitions are properly defined."""
        definitions = builder.get_feature_definitions()

        assert len(definitions) > 0

        # Check for expected features
        feature_names = [d.name for d in definitions]
        assert "grid_cell" in feature_names
        assert "crime_count_7d" in feature_names
        assert "crime_count_30d" in feature_names
        assert "shooting_count_30d" in feature_names
        assert "crime_risk_score" in feature_names

    def test_run_success(self, builder, sample_processed_data):
        """Test successful feature building."""
        result = builder.run(sample_processed_data, execution_date="2024-01-15")

        assert result.success
        assert result.dataset == "crime"
        assert result.rows_output > 0
        assert result.features_computed > 0

    def test_build_features_creates_grid_cells(self, builder, sample_processed_data):
        """Test that grid cells are created."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "grid_cell" in df.columns
        assert df["grid_cell"].notna().all()

    def test_build_features_has_crime_counts(self, builder, sample_processed_data):
        """Test that crime counts are computed."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "crime_count_7d" in df.columns
        assert "crime_count_30d" in df.columns
        assert "crime_count_90d" in df.columns
        assert (df["crime_count_30d"] >= df["crime_count_7d"]).all()

    def test_build_features_has_shooting_features(self, builder, sample_processed_data):
        """Test that shooting features are computed."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "shooting_count_30d" in df.columns
        assert "shooting_ratio_30d" in df.columns
        assert df["shooting_ratio_30d"].between(0, 1).all()

    def test_build_features_has_category_ratios(self, builder, sample_processed_data):
        """Test that category ratios are computed."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "violent_crime_ratio" in df.columns
        assert "property_crime_ratio" in df.columns
        assert df["violent_crime_ratio"].between(0, 1).all()
        assert df["property_crime_ratio"].between(0, 1).all()

    def test_build_features_has_temporal_ratios(self, builder, sample_processed_data):
        """Test that temporal ratios are computed."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "night_crime_ratio" in df.columns
        assert "weekend_crime_ratio" in df.columns
        assert df["night_crime_ratio"].between(0, 1).all()
        assert df["weekend_crime_ratio"].between(0, 1).all()

    def test_build_features_has_risk_score(self, builder, sample_processed_data):
        """Test that risk score is computed."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "crime_risk_score" in df.columns
        assert df["crime_risk_score"].between(0, 1).all()

    def test_build_features_has_grid_coordinates(self, builder, sample_processed_data):
        """Test that grid coordinates are included."""
        builder.run(sample_processed_data, execution_date="2024-01-15")
        df = builder.get_data()

        assert "grid_lat" in df.columns
        assert "grid_long" in df.columns
        assert df["grid_lat"].between(42.2, 42.4).all()
        assert df["grid_long"].between(-71.2, -70.9).all()

    def test_build_features_empty_data(self, builder):
        """Test handling of empty data."""
        df = pd.DataFrame(columns=["lat", "long", "occurred_on_date"])

        result = builder.run(df, execution_date="2024-01-15")

        # Should handle gracefully
        assert result.rows_output == 0

    def test_build_features_no_valid_coords(self, builder):
        """Test handling of data with no valid coordinates."""
        df = pd.DataFrame(
            {
                "incident_number": ["I001"],
                "offense_category": ["Test"],
                "district": ["A1"],
                "shooting": [False],
                "occurred_on_date": [datetime.now()],
                "hour": [14],
                "day_of_week": ["Monday"],
                "lat": [np.nan],
                "long": [np.nan],
            }
        )

        result = builder.run(df, execution_date="2024-01-15")

        # Should handle gracefully with no output
        assert result.rows_output == 0

    def test_feature_stats_computed(self, builder, sample_processed_data):
        """Test that feature statistics are computed."""
        result = builder.run(sample_processed_data, execution_date="2024-01-15")

        assert len(result.feature_stats) > 0


class TestFeatureComputationMethods:
    """Test individual feature computation methods."""

    @pytest.fixture
    def builder(self):
        """Create a CrimeFeatureBuilder instance."""
        return CrimeFeatureBuilder()

    def test_add_grid_cell(self, builder):
        """Test grid cell computation."""
        df = pd.DataFrame(
            {
                "lat": [42.3601, 42.3602, 42.3700],
                "long": [-71.0589, -71.0590, -71.0600],
            }
        )

        df = builder._add_grid_cell(df)

        assert "grid_cell" in df.columns
        assert "grid_lat" in df.columns
        assert "grid_long" in df.columns
        # First two should be in same cell (close together)
        # Third one should be in different cell (farther away)

    def test_compute_category_ratios(self, builder):
        """Test category ratio computation."""
        df = pd.DataFrame(
            {
                "offense_category": ["Assault", "Larceny", "Assault", "Robbery"],
            }
        )

        ratios = builder._compute_category_ratios(df)

        assert "violent_crime_ratio" in ratios
        assert "property_crime_ratio" in ratios
        assert ratios["violent_crime_ratio"] == 0.75  # 3/4 violent (assault, robbery)

    def test_compute_temporal_ratios(self, builder):
        """Test temporal ratio computation."""
        df = pd.DataFrame(
            {
                "hour": [22, 3, 10, 14],  # 2 night (22, 3), 2 day (10, 14)
                "day_of_week": ["Saturday", "Sunday", "Monday", "Tuesday"],  # 2 weekend
            }
        )

        ratios = builder._compute_temporal_ratios(df)

        assert "night_crime_ratio" in ratios
        assert "weekend_crime_ratio" in ratios
        assert ratios["night_crime_ratio"] == 0.5  # 2/4
        assert ratios["weekend_crime_ratio"] == 0.5  # 2/4

    def test_compute_risk_score(self, builder):
        """Test risk score computation."""
        df = pd.DataFrame(
            {
                "crime_count_30d": [10, 50, 100],
                "shooting_count_30d": [0, 1, 5],
                "violent_crime_ratio": [0.1, 0.3, 0.5],
            }
        )

        df = builder._compute_risk_score(df)

        assert "crime_risk_score" in df.columns
        assert df["crime_risk_score"].between(0, 1).all()
        # Higher values should have higher scores
        assert df["crime_risk_score"].iloc[2] > df["crime_risk_score"].iloc[0]


class TestBaseFeatureBuilderMethods:
    """Test inherited methods from BaseFeatureBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a CrimeFeatureBuilder instance."""
        return CrimeFeatureBuilder()

    def test_compute_rolling_count(self, builder):
        """Test rolling count computation."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "A", "B", "B"],
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-05",
                        "2024-01-10",
                        "2024-01-01",
                        "2024-01-02",
                    ]
                ),
            }
        )

        result = builder.compute_rolling_count(
            df,
            group_col="group",
            date_col="date",
            window_days=7,
            output_col="count_7d",
        )

        assert "count_7d" in result.columns

    def test_add_time_features(self, builder):
        """Test time feature extraction."""
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15 14:30:00", "2024-01-20 22:15:00"])})

        df = builder.add_time_features(df, "date")

        assert "hour" in df.columns
        assert "dayofweek" in df.columns
        assert "is_weekend" in df.columns
        assert "is_night" in df.columns

    def test_normalize_feature_minmax(self, builder):
        """Test minmax normalization."""
        df = pd.DataFrame({"value": [0, 50, 100]})

        df = builder.normalize_feature(df, "value", method="minmax")

        assert "value_normalized" in df.columns
        assert df["value_normalized"].min() == 0.0
        assert df["value_normalized"].max() == 1.0

    def test_normalize_feature_zscore(self, builder):
        """Test zscore normalization."""
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})

        df = builder.normalize_feature(df, "value", method="zscore")

        assert "value_normalized" in df.columns
        # Mean should be approximately 0
        assert abs(df["value_normalized"].mean()) < 0.01


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_build_crime_features(self):
        """Test build_crime_features convenience function."""
        base_date = datetime(2024, 1, 15)
        df = pd.DataFrame(
            {
                "incident_number": ["I001", "I002"],
                "offense_category": ["Assault", "Larceny"],
                "district": ["A1", "A1"],
                "shooting": [False, False],
                "occurred_on_date": pd.to_datetime([base_date, base_date - timedelta(days=1)]),
                "hour": [14, 16],
                "day_of_week": ["Monday", "Sunday"],
                "lat": [42.360, 42.360],
                "long": [-71.058, -71.058],
            }
        )

        result = build_crime_features(df, execution_date="2024-01-15")

        assert isinstance(result, dict)
        assert "features_computed" in result or "rows_output" in result
