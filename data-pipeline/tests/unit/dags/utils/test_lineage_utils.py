"""
Tests for Lineage DAG Utilities

Tests for convenience functions used in Airflow DAGs for lineage tracking.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from dags.utils.lineage_utils import (
    compare_runs,
    create_record_lineage_task,
    find_runs_with_schema,
    get_dataset_lineage_history,
    get_lineage_for_date,
    get_restore_commands,
    print_lineage_diff,
    print_lineage_summary,
    print_restore_commands,
    record_pipeline_lineage,
)
from src.shared.lineage import ArtifactVersion, LineageDiff, LineageRecord

# Mock airflow modules before importing DAG utilities
sys.modules["airflow"] = MagicMock()
sys.modules["airflow.models"] = MagicMock()
sys.modules["airflow.providers"] = MagicMock()
sys.modules["airflow.providers.google"] = MagicMock()
sys.modules["airflow.providers.google.cloud"] = MagicMock()
sys.modules["airflow.providers.google.cloud.hooks"] = MagicMock()
sys.modules["airflow.providers.google.cloud.hooks.gcs"] = MagicMock()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock()
    config.storage.buckets.main = "test-bucket"
    config.storage.emulator.enabled = False
    config.storage.paths.raw = "raw"
    config.storage.paths.processed = "processed"
    config.storage.paths.features = "features"
    config.storage.paths.schemas = "schemas"
    config.gcp_project_id = "test-project"
    config.environment = "test"
    return config


@pytest.fixture
def mock_lineage_tracker():
    """Mock LineageTracker for testing."""
    with patch("dags.utils.lineage_utils.LineageTracker") as mock_tracker_class:
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        yield mock_tracker


@pytest.fixture
def sample_lineage_record():
    """Sample LineageRecord for testing."""
    return LineageRecord(
        dataset="crime",
        execution_date="2026-02-21",
        dag_id="crime_pipeline",
        run_id="manual__2026-02-21T10:00:00+00:00",
        data_versions={
            "raw": ArtifactVersion(
                path="gs://test-bucket/raw/crime/dt=2026-02-21/data.parquet",
                generation="1111111111111111",
            ),
            "processed": ArtifactVersion(
                path="gs://test-bucket/processed/crime/dt=2026-02-21/data.parquet",
                generation="2222222222222222",
            ),
        },
        schema_versions={
            "features": ArtifactVersion(
                path="gs://test-bucket/schemas/crime/features/latest.json",
                generation="3333333333333333",
            ),
        },
        rows_ingested=1000,
        rows_processed=950,
        features_generated=25,
        drift_detected=False,
        fairness_passed=True,
    )


@pytest.fixture
def mock_airflow_context():
    """Create mock Airflow context."""
    mock_ti = MagicMock()
    mock_ti.xcom_pull.return_value = None

    return {
        "ds": "2026-02-21",
        "run_id": "manual__2026-02-21T10:00:00+00:00",
        "ti": mock_ti,
    }


# =============================================================================
# record_pipeline_lineage Tests
# =============================================================================


class TestRecordPipelineLineage:
    """Tests for record_pipeline_lineage function."""

    def test_basic_recording(
        self, mock_lineage_tracker, mock_airflow_context, sample_lineage_record
    ):
        """Test basic lineage recording."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        result = record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=mock_airflow_context,
        )

        assert result["lineage_recorded"] is True
        assert result["dataset"] == "crime"
        assert result["execution_date"] == "2026-02-21"
        mock_lineage_tracker.record_lineage.assert_called_once()

    def test_extracts_stats_from_xcom(self, mock_lineage_tracker, sample_lineage_record):
        """Test that stats are extracted from XCom."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda task_ids: {
            "ingest_data": {"rows_fetched": 1000},
            "preprocess_data": {"rows_output": 950},
            "build_features": {"features_computed": 25},
            "detect_drift": {"drift_detected": False},
            "check_fairness": {"passes_fairness_gate": True},
        }.get(task_ids)

        context = {
            "ds": "2026-02-21",
            "run_id": "test_run",
            "ti": mock_ti,
        }

        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=context,
        )

        call_kwargs = mock_lineage_tracker.record_lineage.call_args[1]
        assert call_kwargs["rows_ingested"] == 1000
        assert call_kwargs["rows_processed"] == 950
        assert call_kwargs["features_generated"] == 25
        assert call_kwargs["drift_detected"] is False
        assert call_kwargs["fairness_passed"] is True

    def test_handles_missing_xcom_data(self, mock_lineage_tracker, sample_lineage_record):
        """Test handling when XCom data is missing."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = None

        context = {
            "ds": "2026-02-21",
            "run_id": "test_run",
            "ti": mock_ti,
        }

        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=context,
        )

        call_kwargs = mock_lineage_tracker.record_lineage.call_args[1]
        assert call_kwargs["rows_ingested"] is None
        assert call_kwargs["rows_processed"] is None

    def test_handles_no_task_instance(self, mock_lineage_tracker, sample_lineage_record):
        """Test handling when task instance is not available."""
        context = {
            "ds": "2026-02-21",
            "run_id": "test_run",
        }

        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        result = record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=context,
        )

        assert result["lineage_recorded"] is True
        assert result["rows_ingested"] is None

    def test_returns_version_info(self, mock_lineage_tracker, sample_lineage_record):
        """Test that version info is returned."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        result = record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context={"ds": "2026-02-21", "run_id": "test"},
        )

        assert "data_versions" in result
        assert "schema_versions" in result
        assert result["data_versions"]["raw"] == "1111111111111111"
        assert result["data_versions"]["processed"] == "2222222222222222"

    def test_uses_custom_config(self, mock_config, mock_lineage_tracker, sample_lineage_record):
        """Test that custom config is passed to tracker."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        with patch("dags.utils.lineage_utils.LineageTracker") as mock_class:
            mock_class.return_value = mock_lineage_tracker

            record_pipeline_lineage(
                dataset="crime",
                dag_id="crime_pipeline",
                context={"ds": "2026-02-21"},
                config=mock_config,
            )

            mock_class.assert_called_once_with(mock_config)


# =============================================================================
# create_record_lineage_task Tests
# =============================================================================


class TestCreateRecordLineageTask:
    """Tests for create_record_lineage_task factory function."""

    def test_creates_callable(self):
        """Test that factory creates a callable."""
        task_func = create_record_lineage_task("crime", "crime_pipeline")
        assert callable(task_func)

    def test_callable_calls_record_pipeline_lineage(
        self, mock_lineage_tracker, sample_lineage_record
    ):
        """Test that created callable calls record_pipeline_lineage."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        task_func = create_record_lineage_task("crime", "crime_pipeline")

        context = {"ds": "2026-02-21", "run_id": "test"}
        result = task_func(**context)

        assert result["lineage_recorded"] is True
        assert result["dataset"] == "crime"

    def test_preserves_dataset_and_dag_id(self, mock_lineage_tracker, sample_lineage_record):
        """Test that dataset and dag_id are preserved."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        task_func = create_record_lineage_task("service_311", "311_pipeline")

        context = {"ds": "2026-02-21", "run_id": "test"}
        task_func(**context)

        call_kwargs = mock_lineage_tracker.record_lineage.call_args[1]
        assert call_kwargs["dataset"] == "service_311"
        assert call_kwargs["dag_id"] == "311_pipeline"

    def test_passes_custom_config(self, mock_config, mock_lineage_tracker, sample_lineage_record):
        """Test that custom config is passed through."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        with patch("dags.utils.lineage_utils.LineageTracker") as mock_class:
            mock_class.return_value = mock_lineage_tracker

            task_func = create_record_lineage_task("crime", "crime_pipeline", config=mock_config)
            task_func(ds="2026-02-21")

            mock_class.assert_called_with(mock_config)


# =============================================================================
# Query Function Tests
# =============================================================================


class TestGetLineageForDate:
    """Tests for get_lineage_for_date function."""

    def test_retrieves_lineage(self, mock_lineage_tracker, sample_lineage_record):
        """Test lineage retrieval."""
        mock_lineage_tracker.get_lineage.return_value = sample_lineage_record

        result = get_lineage_for_date("crime", "2026-02-21")

        assert result == sample_lineage_record
        mock_lineage_tracker.get_lineage.assert_called_once_with("crime", "2026-02-21")

    def test_returns_none_when_not_found(self, mock_lineage_tracker):
        """Test returns None when lineage not found."""
        mock_lineage_tracker.get_lineage.return_value = None

        result = get_lineage_for_date("crime", "2026-02-21")

        assert result is None


class TestGetDatasetLineageHistory:
    """Tests for get_dataset_lineage_history function."""

    def test_retrieves_history(self, mock_lineage_tracker, sample_lineage_record):
        """Test history retrieval."""
        mock_lineage_tracker.get_dataset_history.return_value = [sample_lineage_record]

        result = get_dataset_lineage_history("crime", limit=10)

        assert len(result) == 1
        mock_lineage_tracker.get_dataset_history.assert_called_once_with("crime", 10)

    def test_default_limit(self, mock_lineage_tracker):
        """Test default limit is 30."""
        mock_lineage_tracker.get_dataset_history.return_value = []

        get_dataset_lineage_history("crime")

        mock_lineage_tracker.get_dataset_history.assert_called_once_with("crime", 30)


class TestCompareRuns:
    """Tests for compare_runs function."""

    def test_compares_runs(self, mock_lineage_tracker):
        """Test run comparison."""
        mock_diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
            data_changes={"raw": {"status": "changed"}},
        )
        mock_lineage_tracker.compare_lineage.return_value = mock_diff

        result = compare_runs("crime", "2026-02-20", "2026-02-21")

        assert result == mock_diff
        mock_lineage_tracker.compare_lineage.assert_called_once_with(
            "crime", "2026-02-20", "2026-02-21"
        )


class TestFindRunsWithSchema:
    """Tests for find_runs_with_schema function."""

    def test_finds_runs(self, mock_lineage_tracker, sample_lineage_record):
        """Test finding runs with specific schema."""
        mock_lineage_tracker.find_runs_by_schema_generation.return_value = [sample_lineage_record]

        result = find_runs_with_schema("crime", "3333333333333333", "features")

        assert len(result) == 1
        mock_lineage_tracker.find_runs_by_schema_generation.assert_called_once_with(
            "crime", "3333333333333333", "features"
        )

    def test_default_layer(self, mock_lineage_tracker):
        """Test default layer is features."""
        mock_lineage_tracker.find_runs_by_schema_generation.return_value = []

        find_runs_with_schema("crime", "123")

        mock_lineage_tracker.find_runs_by_schema_generation.assert_called_once_with(
            "crime", "123", "features"
        )


# =============================================================================
# Recovery Function Tests
# =============================================================================


class TestGetRestoreCommands:
    """Tests for get_restore_commands function."""

    def test_gets_commands(self, mock_lineage_tracker):
        """Test getting restore commands."""
        mock_commands = {
            "raw": "gsutil cp gs://bucket/raw#123 gs://bucket/raw",
            "processed": "gsutil cp gs://bucket/processed#456 gs://bucket/processed",
        }
        mock_lineage_tracker.get_restore_commands.return_value = mock_commands

        result = get_restore_commands("crime", "2026-02-21")

        assert result == mock_commands
        mock_lineage_tracker.get_restore_commands.assert_called_once_with(
            "crime", "2026-02-21", None
        )

    def test_with_specific_layers(self, mock_lineage_tracker):
        """Test with specific layers."""
        mock_lineage_tracker.get_restore_commands.return_value = {"raw": "gsutil cp ..."}

        get_restore_commands("crime", "2026-02-21", layers=["raw"])

        mock_lineage_tracker.get_restore_commands.assert_called_once_with(
            "crime", "2026-02-21", ["raw"]
        )


class TestPrintRestoreCommands:
    """Tests for print_restore_commands function."""

    def test_prints_commands(self, mock_lineage_tracker, capsys):
        """Test printing restore commands."""
        mock_commands = {
            "raw": "gsutil cp gs://bucket/raw#123 gs://bucket/raw",
            "processed": "gsutil cp gs://bucket/processed#456 gs://bucket/processed",
        }
        mock_lineage_tracker.get_restore_commands.return_value = mock_commands

        print_restore_commands("crime", "2026-02-21")

        captured = capsys.readouterr()
        assert "Restore commands for crime/2026-02-21" in captured.out
        assert "gsutil cp" in captured.out
        assert "# raw" in captured.out
        assert "# processed" in captured.out


# =============================================================================
# Display Function Tests
# =============================================================================


class TestPrintLineageSummary:
    """Tests for print_lineage_summary function."""

    def test_calls_tracker_method(self, mock_lineage_tracker):
        """Test that it calls tracker's print method."""
        print_lineage_summary("crime", "2026-02-21")

        mock_lineage_tracker.print_lineage_summary.assert_called_once_with("crime", "2026-02-21")


class TestPrintLineageDiff:
    """Tests for print_lineage_diff function."""

    def test_prints_diff_with_changes(self, mock_lineage_tracker, capsys):
        """Test printing diff with changes."""
        mock_diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
            data_changes={
                "raw": {"status": "changed", "from": "111", "to": "222"},
                "features": {"status": "added", "to": "333"},
            },
            schema_changes={
                "processed": {"status": "changed", "from": "444", "to": "555"},
            },
            stats_changes={
                "rows_ingested": {"from": 1000, "to": 1200},
            },
        )
        mock_lineage_tracker.compare_lineage.return_value = mock_diff

        print_lineage_diff("crime", "2026-02-20", "2026-02-21")

        captured = capsys.readouterr()
        assert "LINEAGE DIFF: crime" in captured.out
        assert "From: 2026-02-20 → To: 2026-02-21" in captured.out
        assert "Data Changes" in captured.out
        assert "raw:" in captured.out
        assert "111 → 222" in captured.out
        assert "[NEW]" in captured.out
        assert "Schema Changes" in captured.out
        assert "Stats Changes" in captured.out

    def test_prints_diff_no_changes(self, mock_lineage_tracker, capsys):
        """Test printing diff with no changes."""
        mock_diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
        )
        mock_lineage_tracker.compare_lineage.return_value = mock_diff

        print_lineage_diff("crime", "2026-02-20", "2026-02-21")

        captured = capsys.readouterr()
        assert "No changes" in captured.out

    def test_prints_removed_data(self, mock_lineage_tracker, capsys):
        """Test printing removed data changes."""
        mock_diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
            data_changes={
                "mitigated": {"status": "removed", "from": "999"},
            },
        )
        mock_lineage_tracker.compare_lineage.return_value = mock_diff

        print_lineage_diff("crime", "2026-02-20", "2026-02-21")

        captured = capsys.readouterr()
        assert "[REMOVED]" in captured.out


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_record_lineage_with_empty_context(self, mock_lineage_tracker, sample_lineage_record):
        """Test recording with minimal context."""
        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        result = record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context={"ds": "2026-02-21"},
        )

        assert result["lineage_recorded"] is True

    def test_record_lineage_partial_xcom_data(self, mock_lineage_tracker, sample_lineage_record):
        """Test recording with partial XCom data."""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.side_effect = lambda task_ids: {
            "ingest_data": {"rows_fetched": 1000},
        }.get(task_ids)

        context = {
            "ds": "2026-02-21",
            "ti": mock_ti,
        }

        mock_lineage_tracker.record_lineage.return_value = sample_lineage_record

        result = record_pipeline_lineage(
            dataset="crime",
            dag_id="crime_pipeline",
            context=context,
        )

        assert result["rows_ingested"] == 1000
        assert result["rows_processed"] is None

    def test_get_lineage_history_empty(self, mock_lineage_tracker):
        """Test getting empty history."""
        mock_lineage_tracker.get_dataset_history.return_value = []

        result = get_dataset_lineage_history("new_dataset")

        assert result == []

    def test_compare_runs_with_missing_lineage(self, mock_lineage_tracker):
        """Test comparing runs when one is missing."""
        mock_diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
        )
        mock_lineage_tracker.compare_lineage.return_value = mock_diff

        result = compare_runs("crime", "2026-02-20", "2026-02-21")

        assert result.data_changes == {}
        assert result.schema_changes == {}
        assert result.stats_changes == {}

    def test_find_runs_with_schema_no_matches(self, mock_lineage_tracker):
        """Test finding runs with no matches."""
        mock_lineage_tracker.find_runs_by_schema_generation.return_value = []

        result = find_runs_with_schema("crime", "nonexistent_gen")

        assert result == []

    def test_get_restore_commands_no_lineage(self, mock_lineage_tracker):
        """Test getting restore commands when no lineage exists."""
        mock_lineage_tracker.get_restore_commands.return_value = {}

        result = get_restore_commands("crime", "2026-02-21")

        assert result == {}
