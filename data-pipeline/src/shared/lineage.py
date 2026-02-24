"""
Boston Pulse - GCS-Native Lineage Tracking

Provides comprehensive data lineage tracking using GCS generation numbers.
This enables:
- Reproducibility: Know exactly what data a model saw on any date
- Debugging: Compare runs to find what changed when something breaks
- Recovery: Restore data to any previous state with a single command
- Audit Trail: Full history of all pipeline runs with metadata

Architecture:
- Primary storage: GCS (lineage/{dataset}/dt={date}/lineage.json)
- Secondary storage: Firestore (for fast queries, optional)
- Uses GCS generation numbers for immutable versioning

Usage:
    from src.shared.lineage import LineageTracker

    tracker = LineageTracker()

    # Record lineage at end of pipeline
    lineage = tracker.record_lineage(
        dataset="crime",
        execution_date="2026-02-21",
        dag_id="crime_pipeline",
        rows_ingested=1000,
    )

    # Query lineage for debugging
    lineage = tracker.get_lineage("crime", "2026-02-21")
    diff = tracker.compare_lineage("crime", "2026-02-20", "2026-02-21")

    # Restore data
    commands = tracker.get_restore_commands("crime", "2026-02-21")
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from google.cloud import storage
from google.cloud.exceptions import NotFound

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ArtifactVersion:
    """
    Represents a versioned artifact in GCS.

    The generation number is the key - it uniquely identifies
    the exact version of the file at the time of the pipeline run.
    """

    path: str
    generation: str | None = None
    size_bytes: int | None = None
    md5_hash: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactVersion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LineageRecord:
    """
    Complete lineage record for a pipeline run.

    This captures everything needed to:
    1. Reproduce the exact data state
    2. Debug issues by comparing runs
    3. Audit what data was used when
    """

    dataset: str
    execution_date: str
    dag_id: str
    run_id: str | None = None

    # GCS generation numbers for each data layer
    data_versions: dict[str, ArtifactVersion] = field(default_factory=dict)

    # Schema versions (also tracked by generation)
    schema_versions: dict[str, ArtifactVersion] = field(default_factory=dict)

    # Pipeline statistics
    rows_ingested: int | None = None
    rows_processed: int | None = None
    features_generated: int | None = None

    # Quality metrics
    drift_detected: bool | None = None
    fairness_passed: bool | None = None

    # Status and metadata
    status: str = "success"
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    pipeline_completed_at: str | None = None

    # Environment info
    environment: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "dataset": self.dataset,
            "execution_date": self.execution_date,
            "dag_id": self.dag_id,
            "run_id": self.run_id,
            "data_versions": {k: v.to_dict() for k, v in self.data_versions.items()},
            "schema_versions": {k: v.to_dict() for k, v in self.schema_versions.items()},
            "rows_ingested": self.rows_ingested,
            "rows_processed": self.rows_processed,
            "features_generated": self.features_generated,
            "drift_detected": self.drift_detected,
            "fairness_passed": self.fairness_passed,
            "status": self.status,
            "recorded_at": self.recorded_at,
            "pipeline_completed_at": self.pipeline_completed_at,
            "environment": self.environment,
            "git_commit": self.git_commit,
        }
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        """Create from dictionary."""
        data_versions = {
            k: ArtifactVersion.from_dict(v) for k, v in data.get("data_versions", {}).items()
        }
        schema_versions = {
            k: ArtifactVersion.from_dict(v) for k, v in data.get("schema_versions", {}).items()
        }

        return cls(
            dataset=data["dataset"],
            execution_date=data["execution_date"],
            dag_id=data["dag_id"],
            run_id=data.get("run_id"),
            data_versions=data_versions,
            schema_versions=schema_versions,
            rows_ingested=data.get("rows_ingested"),
            rows_processed=data.get("rows_processed"),
            features_generated=data.get("features_generated"),
            drift_detected=data.get("drift_detected"),
            fairness_passed=data.get("fairness_passed"),
            status=data.get("status", "success"),
            recorded_at=data.get("recorded_at", datetime.now(UTC).isoformat()),
            pipeline_completed_at=data.get("pipeline_completed_at"),
            environment=data.get("environment"),
            git_commit=data.get("git_commit"),
        )


@dataclass
class LineageDiff:
    """Difference between two lineage records."""

    date_from: str
    date_to: str
    data_changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =============================================================================
# LineageTracker Class
# =============================================================================


class LineageTracker:
    """
    GCS-native lineage tracking for Boston Pulse pipelines.

    This class is designed to be:
    - Modular: Can be used by any dataset pipeline
    - Scalable: Uses GCS generation numbers (no additional infrastructure)
    - Queryable: Stores in both GCS (durable) and Firestore (fast queries)

    Key Methods:
        record_lineage(): Capture generation numbers for all artifacts
        get_lineage(): Retrieve lineage for a specific date
        compare_lineage(): Diff two runs to see what changed
        get_restore_commands(): Generate gsutil restore commands
    """

    # Standard data layers to track
    DATA_LAYERS = ["raw", "processed", "features", "mitigated"]

    def __init__(self, config: Settings | None = None):
        """
        Initialize LineageTracker.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.bucket_name = self.config.storage.buckets.main

        # Initialize GCS client
        if self.config.storage.emulator.enabled:
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": self.config.storage.emulator.host},
            )
        else:
            self.client = storage.Client(project=self.config.gcp_project_id)

        self.bucket = self.client.bucket(self.bucket_name)

        # Firestore client (optional, for fast queries)
        self._firestore_client = None
        self._firestore_enabled = False
        self._init_firestore()

    def _init_firestore(self) -> None:
        """Initialize Firestore client if available."""
        try:
            from google.cloud import firestore

            self._firestore_client = firestore.Client(project=self.config.gcp_project_id)
            self._firestore_enabled = True
            logger.info("Firestore enabled for lineage queries")
        except Exception as e:
            logger.warning(f"Firestore not available, using GCS-only mode: {e}")
            self._firestore_enabled = False

    # =========================================================================
    # Core Recording Methods
    # =========================================================================

    def record_lineage(
        self,
        dataset: str,
        execution_date: str,
        dag_id: str,
        run_id: str | None = None,
        rows_ingested: int | None = None,
        rows_processed: int | None = None,
        features_generated: int | None = None,
        drift_detected: bool | None = None,
        fairness_passed: bool | None = None,
        status: str = "success",
    ) -> LineageRecord:
        """
        Record lineage for a pipeline run.

        This should be called at the END of the pipeline, after all
        data has been written. It captures the GCS generation numbers
        for all artifacts.

        Args:
            dataset: Dataset name (crime, service_311, etc.)
            execution_date: Execution date in YYYY-MM-DD format
            dag_id: Airflow DAG ID
            run_id: Airflow run ID (optional)
            rows_ingested: Number of rows ingested
            rows_processed: Number of rows after processing
            features_generated: Number of features computed
            drift_detected: Whether drift was detected
            fairness_passed: Whether fairness checks passed
            status: Pipeline status (success, failed, partial)

        Returns:
            LineageRecord with all captured information
        """
        logger.info(f"Recording lineage for {dataset}/{execution_date}")

        # Capture data versions (GCS generation numbers)
        data_versions = self._capture_data_versions(dataset, execution_date)

        # Capture schema versions
        schema_versions = self._capture_schema_versions(dataset)

        # Get git commit if available
        git_commit = self._get_git_commit()

        # Create lineage record
        lineage = LineageRecord(
            dataset=dataset,
            execution_date=execution_date,
            dag_id=dag_id,
            run_id=run_id,
            data_versions=data_versions,
            schema_versions=schema_versions,
            rows_ingested=rows_ingested,
            rows_processed=rows_processed,
            features_generated=features_generated,
            drift_detected=drift_detected,
            fairness_passed=fairness_passed,
            status=status,
            pipeline_completed_at=datetime.now(UTC).isoformat(),
            environment=self.config.environment,
            git_commit=git_commit,
        )

        # Save to GCS (primary storage)
        self._save_lineage_to_gcs(lineage)

        # Save to Firestore (secondary, for fast queries)
        if self._firestore_enabled:
            self._save_lineage_to_firestore(lineage)

        logger.info(
            f"Lineage recorded for {dataset}/{execution_date}: "
            f"{len(data_versions)} data versions, {len(schema_versions)} schema versions"
        )

        return lineage

    def _capture_data_versions(
        self, dataset: str, execution_date: str
    ) -> dict[str, ArtifactVersion]:
        """Capture GCS generation numbers for all data layers."""
        versions = {}

        for layer in self.DATA_LAYERS:
            layer_path = getattr(self.config.storage.paths, layer, layer)
            path = f"{layer_path}/{dataset}/dt={execution_date}/data.parquet"

            try:
                blob = self.bucket.get_blob(path)
                if blob is not None:
                    versions[layer] = ArtifactVersion(
                        path=f"gs://{self.bucket_name}/{path}",
                        generation=str(blob.generation),
                        size_bytes=blob.size,
                        md5_hash=blob.md5_hash,
                        updated_at=blob.updated.isoformat() if blob.updated else None,
                    )
                    logger.debug(f"Captured {layer} version: generation={blob.generation}")
            except Exception as e:
                logger.debug(f"No {layer} data found for {dataset}/{execution_date}: {e}")

        return versions

    def _capture_schema_versions(self, dataset: str) -> dict[str, ArtifactVersion]:
        """Capture GCS generation numbers for schema files."""
        versions = {}
        schema_path = self.config.storage.paths.schemas

        for layer in ["raw", "processed", "features"]:
            path = f"{schema_path}/{dataset}/{layer}/latest.json"

            try:
                blob = self.bucket.get_blob(path)
                if blob is not None:
                    versions[layer] = ArtifactVersion(
                        path=f"gs://{self.bucket_name}/{path}",
                        generation=str(blob.generation),
                        size_bytes=blob.size,
                        md5_hash=blob.md5_hash,
                        updated_at=blob.updated.isoformat() if blob.updated else None,
                    )
            except Exception as e:
                logger.debug(f"No schema found for {dataset}/{layer}: {e}")

        return versions

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None

    def _save_lineage_to_gcs(self, lineage: LineageRecord) -> str:
        """Save lineage record to GCS."""
        path = f"lineage/{lineage.dataset}/dt={lineage.execution_date}/lineage.json"

        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(lineage.to_dict(), indent=2),
            content_type="application/json",
        )

        logger.debug(f"Saved lineage to gs://{self.bucket_name}/{path}")
        return f"gs://{self.bucket_name}/{path}"

    def _save_lineage_to_firestore(self, lineage: LineageRecord) -> None:
        """Save lineage record to Firestore for fast queries."""
        if not self._firestore_enabled or self._firestore_client is None:
            return

        try:
            doc_id = f"{lineage.dataset}_{lineage.execution_date}"
            collection = self._firestore_client.collection("lineage")
            collection.document(doc_id).set(lineage.to_dict())
            logger.debug(f"Saved lineage to Firestore: {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to save lineage to Firestore: {e}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_lineage(self, dataset: str, execution_date: str) -> LineageRecord | None:
        """
        Get lineage record for a specific date.

        Args:
            dataset: Dataset name
            execution_date: Execution date in YYYY-MM-DD format

        Returns:
            LineageRecord or None if not found
        """
        path = f"lineage/{dataset}/dt={execution_date}/lineage.json"

        try:
            blob = self.bucket.blob(path)
            content = blob.download_as_string()
            data = json.loads(content)
            return LineageRecord.from_dict(data)
        except NotFound:
            logger.debug(f"No lineage found for {dataset}/{execution_date}")
            return None
        except Exception as e:
            logger.error(f"Error reading lineage for {dataset}/{execution_date}: {e}")
            return None

    def get_dataset_history(self, dataset: str, limit: int = 30) -> list[LineageRecord]:
        """
        Get recent lineage records for a dataset.

        Args:
            dataset: Dataset name
            limit: Maximum number of records to return

        Returns:
            List of LineageRecord sorted by date (newest first)
        """
        prefix = f"lineage/{dataset}/dt="
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        records = []
        dates = set()

        for blob in blobs:
            if "lineage.json" in blob.name:
                date_part = blob.name.split("dt=")[1].split("/")[0]
                dates.add(date_part)

        for date in sorted(dates, reverse=True)[:limit]:
            record = self.get_lineage(dataset, date)
            if record:
                records.append(record)

        return records

    def find_runs_by_schema_generation(
        self,
        dataset: str,
        schema_generation: str,
        layer: str = "features",
    ) -> list[LineageRecord]:
        """
        Find all runs that used a specific schema version.

        Useful for identifying all runs affected by a bad schema.

        Args:
            dataset: Dataset name
            schema_generation: GCS generation number of the schema
            layer: Schema layer to check

        Returns:
            List of LineageRecord that used this schema
        """
        history = self.get_dataset_history(dataset, limit=365)

        matching = []
        for record in history:
            schema_version = record.schema_versions.get(layer)
            if schema_version and schema_version.generation == schema_generation:
                matching.append(record)

        return matching

    # =========================================================================
    # Comparison Methods
    # =========================================================================

    def compare_lineage(
        self,
        dataset: str,
        date_from: str,
        date_to: str,
    ) -> LineageDiff:
        """
        Compare lineage between two dates.

        This is the key debugging tool - shows exactly what changed
        between two pipeline runs.

        Args:
            dataset: Dataset name
            date_from: Earlier date
            date_to: Later date

        Returns:
            LineageDiff showing all changes
        """
        lineage_from = self.get_lineage(dataset, date_from)
        lineage_to = self.get_lineage(dataset, date_to)

        diff = LineageDiff(date_from=date_from, date_to=date_to)

        if lineage_from is None or lineage_to is None:
            logger.warning(f"Cannot compare: missing lineage for {date_from} or {date_to}")
            return diff

        # Compare data versions
        all_layers = set(lineage_from.data_versions.keys()) | set(lineage_to.data_versions.keys())
        for layer in all_layers:
            v_from = lineage_from.data_versions.get(layer)
            v_to = lineage_to.data_versions.get(layer)

            if v_from is None and v_to is not None:
                diff.data_changes[layer] = {"status": "added", "to": v_to.generation}
            elif v_from is not None and v_to is None:
                diff.data_changes[layer] = {
                    "status": "removed",
                    "from": v_from.generation,
                }
            elif v_from and v_to and v_from.generation != v_to.generation:
                diff.data_changes[layer] = {
                    "status": "changed",
                    "from": v_from.generation,
                    "to": v_to.generation,
                }

        # Compare schema versions
        all_schemas = set(lineage_from.schema_versions.keys()) | set(
            lineage_to.schema_versions.keys()
        )
        for layer in all_schemas:
            v_from = lineage_from.schema_versions.get(layer)
            v_to = lineage_to.schema_versions.get(layer)

            if v_from and v_to and v_from.generation != v_to.generation:
                diff.schema_changes[layer] = {
                    "status": "changed",
                    "from": v_from.generation,
                    "to": v_to.generation,
                }

        # Compare stats
        if lineage_from.rows_ingested != lineage_to.rows_ingested:
            diff.stats_changes["rows_ingested"] = {
                "from": lineage_from.rows_ingested,
                "to": lineage_to.rows_ingested,
            }
        if lineage_from.rows_processed != lineage_to.rows_processed:
            diff.stats_changes["rows_processed"] = {
                "from": lineage_from.rows_processed,
                "to": lineage_to.rows_processed,
            }
        if lineage_from.drift_detected != lineage_to.drift_detected:
            diff.stats_changes["drift_detected"] = {
                "from": lineage_from.drift_detected,
                "to": lineage_to.drift_detected,
            }

        return diff

    # =========================================================================
    # Recovery Methods
    # =========================================================================

    def get_restore_commands(
        self,
        dataset: str,
        execution_date: str,
        layers: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Generate gsutil commands to restore data to a specific state.

        Uses GCS generation numbers to restore exact versions.

        Args:
            dataset: Dataset name
            execution_date: Date to restore to
            layers: Specific layers to restore (default: all)

        Returns:
            Dict mapping layer to gsutil restore command
        """
        lineage = self.get_lineage(dataset, execution_date)
        if lineage is None:
            logger.error(f"No lineage found for {dataset}/{execution_date}")
            return {}

        layers = layers or list(lineage.data_versions.keys())
        commands = {}

        for layer in layers:
            version = lineage.data_versions.get(layer)
            if version and version.generation:
                source = f'"{version.path}#{version.generation}"'
                dest = version.path
                commands[layer] = f"gsutil cp {source} {dest}"

        return commands

    def restore_artifact(
        self,
        dataset: str,
        layer: str,
        execution_date: str,
        dry_run: bool = True,
    ) -> str | None:
        """
        Restore a specific artifact to its state on a given date.

        Args:
            dataset: Dataset name
            layer: Data layer to restore
            execution_date: Date to restore to
            dry_run: If True, only return command without executing

        Returns:
            Path to restored artifact, or None if failed
        """
        commands = self.get_restore_commands(dataset, execution_date, [layer])

        if layer not in commands:
            logger.error(f"No restore command for {layer}")
            return None

        command = commands[layer]
        logger.info(f"Restore command: {command}")

        if dry_run:
            return command

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                lineage = self.get_lineage(dataset, execution_date)
                if lineage:
                    return lineage.data_versions[layer].path
            else:
                logger.error(f"Restore failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Restore error: {e}")

        return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_lineage_summary(self, dataset: str, execution_date: str) -> None:
        """Print a human-readable lineage summary."""
        lineage = self.get_lineage(dataset, execution_date)

        if lineage is None:
            print(f"No lineage found for {dataset}/{execution_date}")
            return

        print(f"\n{'='*60}")
        print(f"LINEAGE: {dataset} / {execution_date}")
        print(f"{'='*60}")
        print(f"DAG: {lineage.dag_id}")
        print(f"Run ID: {lineage.run_id or 'N/A'}")
        print(f"Status: {lineage.status}")
        print(f"Recorded: {lineage.recorded_at}")
        print(f"Git Commit: {lineage.git_commit or 'N/A'}")

        print("\n--- Data Versions ---")
        for layer, version in lineage.data_versions.items():
            print(f"  {layer}: generation={version.generation}")
            if version.size_bytes:
                print(f"    size={version.size_bytes:,} bytes")

        print("\n--- Schema Versions ---")
        for layer, version in lineage.schema_versions.items():
            print(f"  {layer}: generation={version.generation}")

        print("\n--- Statistics ---")
        print(f"  Rows ingested: {lineage.rows_ingested or 'N/A'}")
        print(f"  Rows processed: {lineage.rows_processed or 'N/A'}")
        print(f"  Features generated: {lineage.features_generated or 'N/A'}")
        print(f"  Drift detected: {lineage.drift_detected}")
        print(f"  Fairness passed: {lineage.fairness_passed}")
        print(f"{'='*60}\n")


# =============================================================================
# Factory Function
# =============================================================================


def get_lineage_tracker(config: Settings | None = None) -> LineageTracker:
    """
    Get a LineageTracker instance.

    This is the recommended way to get a tracker in DAG code.

    Args:
        config: Optional configuration object

    Returns:
        LineageTracker instance
    """
    return LineageTracker(config)
