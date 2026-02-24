"""
Tests for GCS-Native Lineage Tracking

Comprehensive unit tests for:
- ArtifactVersion dataclass
- LineageRecord dataclass
- LineageDiff dataclass
- LineageTracker class methods
"""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.shared.lineage import (
    ArtifactVersion,
    LineageDiff,
    LineageRecord,
    LineageTracker,
    get_lineage_tracker,
)

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
def mock_gcs_client():
    """Mock GCS client and bucket."""
    with patch("src.shared.lineage.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        yield mock_client, mock_bucket


@pytest.fixture
def mock_firestore():
    """Mock Firestore client."""
    with patch("src.shared.lineage.firestore.Client") as mock_fs:
        yield mock_fs


@pytest.fixture
def sample_artifact_version():
    """Sample ArtifactVersion for testing."""
    return ArtifactVersion(
        path="gs://test-bucket/raw/crime/dt=2026-02-21/data.parquet",
        generation="1234567890123456",
        size_bytes=1024000,
        md5_hash="abc123def456",  # pragma: allowlist secret
        updated_at="2026-02-21T10:30:00+00:00",
    )


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
            "features": ArtifactVersion(
                path="gs://test-bucket/features/crime/dt=2026-02-21/data.parquet",
                generation="3333333333333333",
            ),
        },
        schema_versions={
            "raw": ArtifactVersion(
                path="gs://test-bucket/schemas/crime/raw/latest.json",
                generation="4444444444444444",
            ),
        },
        rows_ingested=1000,
        rows_processed=950,
        features_generated=25,
        drift_detected=False,
        fairness_passed=True,
        status="success",
        environment="test",
        git_commit="abc12345",
    )


# =============================================================================
# ArtifactVersion Tests
# =============================================================================


class TestArtifactVersion:
    """Tests for ArtifactVersion dataclass."""

    def test_creation_with_all_fields(self, sample_artifact_version):
        """Test ArtifactVersion creation with all fields."""
        av = sample_artifact_version
        assert av.path == "gs://test-bucket/raw/crime/dt=2026-02-21/data.parquet"
        assert av.generation == "1234567890123456"
        assert av.size_bytes == 1024000
        assert av.md5_hash == "abc123def456"  # pragma: allowlist secret
        assert av.updated_at == "2026-02-21T10:30:00+00:00"

    def test_creation_with_minimal_fields(self):
        """Test ArtifactVersion creation with only required fields."""
        av = ArtifactVersion(path="gs://bucket/path/file.parquet")
        assert av.path == "gs://bucket/path/file.parquet"
        assert av.generation is None
        assert av.size_bytes is None
        assert av.md5_hash is None
        assert av.updated_at is None

    def test_to_dict_excludes_none_values(self):
        """Test that to_dict excludes None values."""
        av = ArtifactVersion(
            path="gs://bucket/file.parquet",
            generation="123",
        )
        result = av.to_dict()
        assert "path" in result
        assert "generation" in result
        assert "size_bytes" not in result
        assert "md5_hash" not in result
        assert "updated_at" not in result

    def test_to_dict_includes_all_non_none_values(self, sample_artifact_version):
        """Test that to_dict includes all non-None values."""
        result = sample_artifact_version.to_dict()
        assert result["path"] == sample_artifact_version.path
        assert result["generation"] == sample_artifact_version.generation
        assert result["size_bytes"] == sample_artifact_version.size_bytes
        assert result["md5_hash"] == sample_artifact_version.md5_hash
        assert result["updated_at"] == sample_artifact_version.updated_at

    def test_from_dict_creates_instance(self):
        """Test from_dict creates correct instance."""
        data = {
            "path": "gs://bucket/file.parquet",
            "generation": "999",
            "size_bytes": 500,
        }
        av = ArtifactVersion.from_dict(data)
        assert av.path == "gs://bucket/file.parquet"
        assert av.generation == "999"
        assert av.size_bytes == 500
        assert av.md5_hash is None

    def test_from_dict_ignores_extra_fields(self):
        """Test from_dict ignores fields not in dataclass."""
        data = {
            "path": "gs://bucket/file.parquet",
            "extra_field": "should_be_ignored",
            "another_extra": 123,
        }
        av = ArtifactVersion.from_dict(data)
        assert av.path == "gs://bucket/file.parquet"
        assert not hasattr(av, "extra_field")

    def test_roundtrip_serialization(self, sample_artifact_version):
        """Test to_dict -> from_dict roundtrip preserves data."""
        dict_repr = sample_artifact_version.to_dict()
        restored = ArtifactVersion.from_dict(dict_repr)
        assert restored.path == sample_artifact_version.path
        assert restored.generation == sample_artifact_version.generation
        assert restored.size_bytes == sample_artifact_version.size_bytes
        assert restored.md5_hash == sample_artifact_version.md5_hash


# =============================================================================
# LineageRecord Tests
# =============================================================================


class TestLineageRecord:
    """Tests for LineageRecord dataclass."""

    def test_creation_with_required_fields(self):
        """Test LineageRecord creation with only required fields."""
        lr = LineageRecord(
            dataset="crime",
            execution_date="2026-02-21",
            dag_id="crime_pipeline",
        )
        assert lr.dataset == "crime"
        assert lr.execution_date == "2026-02-21"
        assert lr.dag_id == "crime_pipeline"
        assert lr.status == "success"
        assert lr.data_versions == {}
        assert lr.schema_versions == {}

    def test_creation_with_all_fields(self, sample_lineage_record):
        """Test LineageRecord creation with all fields."""
        lr = sample_lineage_record
        assert lr.dataset == "crime"
        assert lr.rows_ingested == 1000
        assert lr.rows_processed == 950
        assert lr.features_generated == 25
        assert lr.drift_detected is False
        assert lr.fairness_passed is True
        assert lr.git_commit == "abc12345"
        assert len(lr.data_versions) == 3
        assert len(lr.schema_versions) == 1

    def test_to_dict_structure(self, sample_lineage_record):
        """Test to_dict produces correct structure."""
        result = sample_lineage_record.to_dict()
        assert result["dataset"] == "crime"
        assert result["execution_date"] == "2026-02-21"
        assert result["dag_id"] == "crime_pipeline"
        assert "data_versions" in result
        assert "schema_versions" in result
        assert isinstance(result["data_versions"]["raw"], dict)

    def test_to_dict_excludes_none_values(self):
        """Test to_dict excludes None values."""
        lr = LineageRecord(
            dataset="test",
            execution_date="2026-02-21",
            dag_id="test_dag",
        )
        result = lr.to_dict()
        assert "run_id" not in result
        assert "rows_ingested" not in result
        assert "git_commit" not in result

    def test_from_dict_creates_instance(self):
        """Test from_dict creates correct instance."""
        data = {
            "dataset": "crime",
            "execution_date": "2026-02-21",
            "dag_id": "crime_pipeline",
            "rows_ingested": 500,
            "data_versions": {
                "raw": {
                    "path": "gs://bucket/raw/crime/dt=2026-02-21/data.parquet",
                    "generation": "111",
                }
            },
            "schema_versions": {},
        }
        lr = LineageRecord.from_dict(data)
        assert lr.dataset == "crime"
        assert lr.rows_ingested == 500
        assert "raw" in lr.data_versions
        assert isinstance(lr.data_versions["raw"], ArtifactVersion)

    def test_from_dict_handles_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "dataset": "test",
            "execution_date": "2026-02-21",
            "dag_id": "test_dag",
        }
        lr = LineageRecord.from_dict(data)
        assert lr.run_id is None
        assert lr.rows_ingested is None
        assert lr.drift_detected is None
        assert lr.data_versions == {}
        assert lr.schema_versions == {}

    def test_roundtrip_serialization(self, sample_lineage_record):
        """Test to_dict -> from_dict roundtrip preserves data."""
        dict_repr = sample_lineage_record.to_dict()
        restored = LineageRecord.from_dict(dict_repr)
        assert restored.dataset == sample_lineage_record.dataset
        assert restored.execution_date == sample_lineage_record.execution_date
        assert restored.rows_ingested == sample_lineage_record.rows_ingested
        assert len(restored.data_versions) == len(sample_lineage_record.data_versions)
        assert (
            restored.data_versions["raw"].generation
            == sample_lineage_record.data_versions["raw"].generation
        )

    def test_json_serialization(self, sample_lineage_record):
        """Test LineageRecord can be serialized to JSON."""
        dict_repr = sample_lineage_record.to_dict()
        json_str = json.dumps(dict_repr)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["dataset"] == "crime"

    def test_recorded_at_default_value(self):
        """Test recorded_at gets default timestamp."""
        lr = LineageRecord(
            dataset="test",
            execution_date="2026-02-21",
            dag_id="test_dag",
        )
        assert lr.recorded_at is not None
        assert "T" in lr.recorded_at  # ISO format


# =============================================================================
# LineageDiff Tests
# =============================================================================


class TestLineageDiff:
    """Tests for LineageDiff dataclass."""

    def test_creation_empty_diff(self):
        """Test LineageDiff creation with no changes."""
        diff = LineageDiff(date_from="2026-02-20", date_to="2026-02-21")
        assert diff.date_from == "2026-02-20"
        assert diff.date_to == "2026-02-21"
        assert diff.data_changes == {}
        assert diff.schema_changes == {}
        assert diff.stats_changes == {}

    def test_creation_with_changes(self):
        """Test LineageDiff creation with changes."""
        diff = LineageDiff(
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
        assert len(diff.data_changes) == 2
        assert len(diff.schema_changes) == 1
        assert len(diff.stats_changes) == 1

    def test_to_dict(self):
        """Test to_dict produces correct structure."""
        diff = LineageDiff(
            date_from="2026-02-20",
            date_to="2026-02-21",
            data_changes={"raw": {"status": "changed"}},
        )
        result = diff.to_dict()
        assert result["date_from"] == "2026-02-20"
        assert result["date_to"] == "2026-02-21"
        assert "data_changes" in result
        assert result["data_changes"]["raw"]["status"] == "changed"


# =============================================================================
# LineageTracker Tests
# =============================================================================


class TestLineageTracker:
    """Tests for LineageTracker class."""

    def test_initialization(self, mock_config, mock_gcs_client):
        """Test LineageTracker initialization."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        assert tracker.bucket_name == "test-bucket"
        assert tracker.config == mock_config

    def test_initialization_with_emulator(self, mock_config, mock_gcs_client):
        """Test LineageTracker initialization with storage emulator."""
        mock_config.storage.emulator.enabled = True
        mock_config.storage.emulator.host = "http://localhost:4443"
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            LineageTracker(mock_config)

        mock_client.assert_called_with(
            project="test-project",
            client_options={"api_endpoint": "http://localhost:4443"},
        )

    def test_data_layers_constant(self):
        """Test DATA_LAYERS constant is defined correctly."""
        assert "raw" in LineageTracker.DATA_LAYERS
        assert "processed" in LineageTracker.DATA_LAYERS
        assert "features" in LineageTracker.DATA_LAYERS
        assert "mitigated" in LineageTracker.DATA_LAYERS

    def test_capture_data_versions(self, mock_config, mock_gcs_client):
        """Test _capture_data_versions captures GCS generations."""
        mock_client, mock_bucket = mock_gcs_client

        mock_blob = MagicMock()
        mock_blob.generation = 1234567890123456
        mock_blob.size = 1024
        mock_blob.md5_hash = "abc123"
        mock_blob.updated = datetime.now(UTC)
        mock_bucket.get_blob.return_value = mock_blob

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        versions = tracker._capture_data_versions("crime", "2026-02-21")

        assert "raw" in versions
        assert versions["raw"].generation == "1234567890123456"
        assert versions["raw"].size_bytes == 1024

    def test_capture_data_versions_missing_layer(self, mock_config, mock_gcs_client):
        """Test _capture_data_versions handles missing layers."""
        mock_client, mock_bucket = mock_gcs_client
        mock_bucket.get_blob.return_value = None

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        versions = tracker._capture_data_versions("crime", "2026-02-21")

        assert versions == {}

    def test_capture_schema_versions(self, mock_config, mock_gcs_client):
        """Test _capture_schema_versions captures schema generations."""
        mock_client, mock_bucket = mock_gcs_client

        mock_blob = MagicMock()
        mock_blob.generation = 9999999999999999
        mock_blob.size = 512
        mock_blob.md5_hash = "schema123"
        mock_blob.updated = datetime.now(UTC)
        mock_bucket.get_blob.return_value = mock_blob

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        versions = tracker._capture_schema_versions("crime")

        assert "raw" in versions
        assert "processed" in versions
        assert "features" in versions

    def test_get_git_commit_success(self, mock_config, mock_gcs_client):
        """Test _get_git_commit returns commit hash."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="abc123def456789\n",
            )
            commit = tracker._get_git_commit()

        assert commit == "abc123de"  # First 8 chars

    def test_get_git_commit_failure(self, mock_config, mock_gcs_client):
        """Test _get_git_commit returns None on failure."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            commit = tracker._get_git_commit()

        assert commit is None

    def test_get_git_commit_exception(self, mock_config, mock_gcs_client):
        """Test _get_git_commit handles exceptions gracefully."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Git not found")
            commit = tracker._get_git_commit()

        assert commit is None

    def test_save_lineage_to_gcs(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test _save_lineage_to_gcs uploads correctly."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        path = tracker._save_lineage_to_gcs(sample_lineage_record)

        mock_bucket.blob.assert_called_once()
        mock_blob.upload_from_string.assert_called_once()
        assert "lineage/crime/dt=2026-02-21/lineage.json" in path

    def test_save_lineage_to_firestore(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test _save_lineage_to_firestore saves correctly."""
        mock_client, mock_bucket = mock_gcs_client
        mock_fs_client = MagicMock()
        mock_collection = MagicMock()
        mock_fs_client.collection.return_value = mock_collection

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker._firestore_client = mock_fs_client
            tracker._firestore_enabled = True

        tracker._save_lineage_to_firestore(sample_lineage_record)

        mock_fs_client.collection.assert_called_with("lineage")
        mock_collection.document.assert_called_with("crime_2026-02-21")

    def test_save_lineage_to_firestore_disabled(
        self, mock_config, mock_gcs_client, sample_lineage_record
    ):
        """Test _save_lineage_to_firestore does nothing when disabled."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker._firestore_enabled = False
            tracker._firestore_client = None

        # Should not raise
        tracker._save_lineage_to_firestore(sample_lineage_record)

    def test_record_lineage(self, mock_config, mock_gcs_client):
        """Test record_lineage creates complete lineage record."""
        mock_client, mock_bucket = mock_gcs_client

        mock_blob = MagicMock()
        mock_blob.generation = 1111111111111111
        mock_blob.size = 1024
        mock_blob.md5_hash = "hash123"
        mock_blob.updated = datetime.now(UTC)
        mock_bucket.get_blob.return_value = mock_blob
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket
            tracker._firestore_enabled = False

        with patch.object(tracker, "_get_git_commit", return_value="abc12345"):
            lineage = tracker.record_lineage(
                dataset="crime",
                execution_date="2026-02-21",
                dag_id="crime_pipeline",
                run_id="test_run_123",
                rows_ingested=1000,
                rows_processed=950,
                features_generated=25,
                drift_detected=False,
                fairness_passed=True,
            )

        assert lineage.dataset == "crime"
        assert lineage.execution_date == "2026-02-21"
        assert lineage.dag_id == "crime_pipeline"
        assert lineage.rows_ingested == 1000
        assert lineage.git_commit == "abc12345"
        assert lineage.environment == "test"

    def test_get_lineage_success(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test get_lineage retrieves and parses lineage."""
        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.download_as_string.return_value = json.dumps(
            sample_lineage_record.to_dict()
        ).encode()
        mock_bucket.blob.return_value = mock_blob

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        lineage = tracker.get_lineage("crime", "2026-02-21")

        assert lineage is not None
        assert lineage.dataset == "crime"
        assert lineage.execution_date == "2026-02-21"

    def test_get_lineage_not_found(self, mock_config, mock_gcs_client):
        """Test get_lineage returns None when not found."""
        from google.cloud.exceptions import NotFound

        mock_client, mock_bucket = mock_gcs_client
        mock_blob = MagicMock()
        mock_blob.download_as_string.side_effect = NotFound("Not found")
        mock_bucket.blob.return_value = mock_blob

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket

        lineage = tracker.get_lineage("crime", "2026-02-21")

        assert lineage is None

    def test_get_dataset_history(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test get_dataset_history returns sorted records."""
        mock_client, mock_bucket = mock_gcs_client

        mock_blob1 = MagicMock()
        mock_blob1.name = "lineage/crime/dt=2026-02-20/lineage.json"
        mock_blob2 = MagicMock()
        mock_blob2.name = "lineage/crime/dt=2026-02-21/lineage.json"

        mock_client.return_value.list_blobs.return_value = [mock_blob1, mock_blob2]

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.client = mock_client.return_value

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            history = tracker.get_dataset_history("crime", limit=10)

        assert len(history) == 2

    def test_compare_lineage_with_changes(self, mock_config, mock_gcs_client):
        """Test compare_lineage detects changes between runs."""
        mock_client, mock_bucket = mock_gcs_client

        lineage_from = LineageRecord(
            dataset="crime",
            execution_date="2026-02-20",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(path="gs://bucket/raw", generation="111"),
                "processed": ArtifactVersion(path="gs://bucket/processed", generation="222"),
            },
            rows_ingested=1000,
            drift_detected=False,
        )

        lineage_to = LineageRecord(
            dataset="crime",
            execution_date="2026-02-21",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(path="gs://bucket/raw", generation="333"),
                "processed": ArtifactVersion(path="gs://bucket/processed", generation="222"),
                "features": ArtifactVersion(path="gs://bucket/features", generation="444"),
            },
            rows_ingested=1200,
            drift_detected=True,
        )

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage") as mock_get:
            mock_get.side_effect = [lineage_from, lineage_to]
            diff = tracker.compare_lineage("crime", "2026-02-20", "2026-02-21")

        assert diff.date_from == "2026-02-20"
        assert diff.date_to == "2026-02-21"
        assert "raw" in diff.data_changes
        assert diff.data_changes["raw"]["status"] == "changed"
        assert "features" in diff.data_changes
        assert diff.data_changes["features"]["status"] == "added"
        assert "rows_ingested" in diff.stats_changes
        assert "drift_detected" in diff.stats_changes

    def test_compare_lineage_no_changes(self, mock_config, mock_gcs_client):
        """Test compare_lineage with identical records."""
        mock_client, mock_bucket = mock_gcs_client

        lineage = LineageRecord(
            dataset="crime",
            execution_date="2026-02-20",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(path="gs://bucket/raw", generation="111"),
            },
            rows_ingested=1000,
        )

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=lineage):
            diff = tracker.compare_lineage("crime", "2026-02-20", "2026-02-21")

        assert diff.data_changes == {}
        assert diff.stats_changes == {}

    def test_compare_lineage_missing_record(self, mock_config, mock_gcs_client):
        """Test compare_lineage handles missing records."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=None):
            diff = tracker.compare_lineage("crime", "2026-02-20", "2026-02-21")

        assert diff.data_changes == {}
        assert diff.schema_changes == {}
        assert diff.stats_changes == {}

    def test_get_restore_commands(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test get_restore_commands generates correct gsutil commands."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            commands = tracker.get_restore_commands("crime", "2026-02-21")

        assert "raw" in commands
        assert "processed" in commands
        assert "features" in commands
        assert "gsutil cp" in commands["raw"]
        assert "#1111111111111111" in commands["raw"]

    def test_get_restore_commands_specific_layers(
        self, mock_config, mock_gcs_client, sample_lineage_record
    ):
        """Test get_restore_commands with specific layers."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            commands = tracker.get_restore_commands("crime", "2026-02-21", layers=["raw"])

        assert "raw" in commands
        assert "processed" not in commands
        assert "features" not in commands

    def test_get_restore_commands_no_lineage(self, mock_config, mock_gcs_client):
        """Test get_restore_commands returns empty dict when no lineage."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=None):
            commands = tracker.get_restore_commands("crime", "2026-02-21")

        assert commands == {}

    def test_restore_artifact_dry_run(self, mock_config, mock_gcs_client, sample_lineage_record):
        """Test restore_artifact in dry run mode."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            result = tracker.restore_artifact("crime", "raw", "2026-02-21", dry_run=True)

        assert result is not None
        assert "gsutil cp" in result

    def test_restore_artifact_missing_layer(
        self, mock_config, mock_gcs_client, sample_lineage_record
    ):
        """Test restore_artifact with missing layer."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            result = tracker.restore_artifact("crime", "nonexistent", "2026-02-21", dry_run=True)

        assert result is None

    def test_find_runs_by_schema_generation(self, mock_config, mock_gcs_client):
        """Test find_runs_by_schema_generation finds matching runs."""
        mock_client, mock_bucket = mock_gcs_client

        record1 = LineageRecord(
            dataset="crime",
            execution_date="2026-02-20",
            dag_id="crime_pipeline",
            schema_versions={
                "features": ArtifactVersion(path="gs://bucket/schema", generation="999"),
            },
        )
        record2 = LineageRecord(
            dataset="crime",
            execution_date="2026-02-21",
            dag_id="crime_pipeline",
            schema_versions={
                "features": ArtifactVersion(path="gs://bucket/schema", generation="999"),
            },
        )
        record3 = LineageRecord(
            dataset="crime",
            execution_date="2026-02-22",
            dag_id="crime_pipeline",
            schema_versions={
                "features": ArtifactVersion(path="gs://bucket/schema", generation="888"),
            },
        )

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_dataset_history", return_value=[record1, record2, record3]):
            matching = tracker.find_runs_by_schema_generation("crime", "999", "features")

        assert len(matching) == 2
        assert all(r.schema_versions["features"].generation == "999" for r in matching)

    def test_print_lineage_summary_with_data(
        self, mock_config, mock_gcs_client, sample_lineage_record, capsys
    ):
        """Test print_lineage_summary outputs correct format."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=sample_lineage_record):
            tracker.print_lineage_summary("crime", "2026-02-21")

        captured = capsys.readouterr()
        assert "LINEAGE: crime / 2026-02-21" in captured.out
        assert "DAG: crime_pipeline" in captured.out
        assert "Data Versions" in captured.out
        assert "Schema Versions" in captured.out
        assert "Statistics" in captured.out

    def test_print_lineage_summary_no_data(self, mock_config, mock_gcs_client, capsys):
        """Test print_lineage_summary handles missing lineage."""
        mock_client, mock_bucket = mock_gcs_client

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=None):
            tracker.print_lineage_summary("crime", "2026-02-21")

        captured = capsys.readouterr()
        assert "No lineage found" in captured.out


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetLineageTracker:
    """Tests for get_lineage_tracker factory function."""

    def test_get_lineage_tracker_default_config(self, mock_gcs_client):
        """Test get_lineage_tracker with default config."""
        mock_client, mock_bucket = mock_gcs_client

        with patch("src.shared.lineage.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.storage.buckets.main = "test-bucket"
            mock_config.storage.emulator.enabled = False
            mock_config.gcp_project_id = "test-project"
            mock_get_config.return_value = mock_config

            with patch.object(LineageTracker, "_init_firestore"):
                tracker = get_lineage_tracker()

        assert isinstance(tracker, LineageTracker)

    def test_get_lineage_tracker_custom_config(self, mock_config, mock_gcs_client):
        """Test get_lineage_tracker with custom config."""
        mock_client, mock_bucket = mock_gcs_client

        with patch.object(LineageTracker, "_init_firestore"):
            tracker = get_lineage_tracker(mock_config)

        assert tracker.config == mock_config


# =============================================================================
# Integration-Style Tests (with mocks)
# =============================================================================


class TestLineageTrackerIntegration:
    """Integration-style tests for LineageTracker workflows."""

    def test_full_lineage_workflow(self, mock_config, mock_gcs_client):
        """Test complete lineage recording and retrieval workflow."""
        mock_client, mock_bucket = mock_gcs_client

        mock_blob = MagicMock()
        mock_blob.generation = 1234567890123456
        mock_blob.size = 2048
        mock_blob.md5_hash = "workflow_hash"
        mock_blob.updated = datetime.now(UTC)
        mock_bucket.get_blob.return_value = mock_blob
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)
            tracker.bucket = mock_bucket
            tracker._firestore_enabled = False

        with patch.object(tracker, "_get_git_commit", return_value="workflow1"):
            lineage = tracker.record_lineage(
                dataset="crime",
                execution_date="2026-02-21",
                dag_id="crime_pipeline",
                rows_ingested=5000,
                rows_processed=4800,
            )

        assert lineage.dataset == "crime"
        assert lineage.rows_ingested == 5000
        assert lineage.rows_processed == 4800
        assert lineage.git_commit == "workflow1"
        assert len(lineage.data_versions) > 0

    def test_lineage_comparison_workflow(self, mock_config, mock_gcs_client):
        """Test lineage comparison workflow."""
        mock_client, mock_bucket = mock_gcs_client

        day1 = LineageRecord(
            dataset="crime",
            execution_date="2026-02-20",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(path="gs://bucket/raw", generation="100"),
                "processed": ArtifactVersion(path="gs://bucket/processed", generation="200"),
            },
            schema_versions={
                "features": ArtifactVersion(path="gs://bucket/schema", generation="300"),
            },
            rows_ingested=1000,
            rows_processed=950,
            drift_detected=False,
        )

        day2 = LineageRecord(
            dataset="crime",
            execution_date="2026-02-21",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(path="gs://bucket/raw", generation="101"),
                "processed": ArtifactVersion(path="gs://bucket/processed", generation="201"),
                "features": ArtifactVersion(path="gs://bucket/features", generation="301"),
            },
            schema_versions={
                "features": ArtifactVersion(path="gs://bucket/schema", generation="300"),
            },
            rows_ingested=1100,
            rows_processed=1050,
            drift_detected=True,
        )

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage") as mock_get:
            mock_get.side_effect = [day1, day2]
            diff = tracker.compare_lineage("crime", "2026-02-20", "2026-02-21")

        assert diff.data_changes["raw"]["status"] == "changed"
        assert diff.data_changes["processed"]["status"] == "changed"
        assert diff.data_changes["features"]["status"] == "added"
        assert diff.schema_changes == {}
        assert diff.stats_changes["rows_ingested"]["from"] == 1000
        assert diff.stats_changes["rows_ingested"]["to"] == 1100
        assert diff.stats_changes["drift_detected"]["from"] is False
        assert diff.stats_changes["drift_detected"]["to"] is True

    def test_restore_workflow(self, mock_config, mock_gcs_client):
        """Test data restoration workflow."""
        mock_client, mock_bucket = mock_gcs_client

        lineage = LineageRecord(
            dataset="crime",
            execution_date="2026-02-21",
            dag_id="crime_pipeline",
            data_versions={
                "raw": ArtifactVersion(
                    path="gs://test-bucket/raw/crime/dt=2026-02-21/data.parquet",
                    generation="restore_gen_123",
                ),
                "processed": ArtifactVersion(
                    path="gs://test-bucket/processed/crime/dt=2026-02-21/data.parquet",
                    generation="restore_gen_456",
                ),
            },
        )

        with (
            patch("src.shared.lineage.get_config", return_value=mock_config),
            patch.object(LineageTracker, "_init_firestore"),
        ):
            tracker = LineageTracker(mock_config)

        with patch.object(tracker, "get_lineage", return_value=lineage):
            commands = tracker.get_restore_commands("crime", "2026-02-21")

        assert "raw" in commands
        assert "processed" in commands
        assert "restore_gen_123" in commands["raw"]
        assert "restore_gen_456" in commands["processed"]
        assert all("gsutil cp" in cmd for cmd in commands.values())
