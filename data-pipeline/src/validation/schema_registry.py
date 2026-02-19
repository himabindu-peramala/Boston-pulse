"""
Boston Pulse - Schema Registry

GCS-backed schema storage with versioning. Manages schema lifecycle:
- Upload schemas to GCS with versioning
- Download schemas by version or get latest
- Schema evolution tracking
- Schema validation against actual data

Usage:
    registry = SchemaRegistry(config)

    # Register a new schema
    registry.register_schema("crime", "raw", schema_dict, version="v1")

    # Get latest schema
    schema = registry.get_schema("crime", "raw")

    # Get specific version
    schema = registry.get_schema("crime", "raw", version="v1")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage

from src.shared.config import Settings, get_config

logger = logging.getLogger(__name__)


@dataclass
class SchemaMetadata:
    """Schema metadata."""

    dataset: str
    layer: str  # raw, processed, features
    version: str
    created_at: datetime
    created_by: str
    description: str | None = None
    num_columns: int = 0
    primary_key: str | None = None


class SchemaRegistry:
    """
    GCS-backed schema registry with versioning.

    Schemas are stored in GCS at:
        gs://{bucket}/schemas/{dataset}/{layer}/{version}.json
        gs://{bucket}/schemas/{dataset}/{layer}/latest.json (symlink)
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize schema registry.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.bucket_name = self.config.storage.buckets.main
        self.schemas_path = self.config.storage.paths.schemas

        # Initialize GCS client
        if self.config.storage.emulator.enabled:
            # Use fake-gcs-server for local development
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": self.config.storage.emulator.host},
            )
        else:
            self.client = storage.Client(project=self.config.gcp_project_id)

        self.bucket = self.client.bucket(self.bucket_name)

    def register_schema(
        self,
        dataset: str,
        layer: str,
        schema: dict[str, Any] | pd.DataFrame,
        version: str | None = None,
        description: str | None = None,
        created_by: str = "system",
        primary_key: str | None = None,
    ) -> str:
        """
        Register a schema in GCS.

        Args:
            dataset: Dataset name (crime, service_311, etc.)
            layer: Data layer (raw, processed, features)
            schema: Schema as dict or pandas DataFrame (will extract schema)
            version: Version string (auto-generated if not provided)
            description: Schema description
            created_by: Creator identifier
            primary_key: Primary key column name

        Returns:
            Version string of the registered schema

        Example:
            schema = {
                "incident_number": {"type": "string", "nullable": False},
                "occurred_date": {"type": "datetime", "nullable": False},
                "latitude": {"type": "float", "nullable": True},
            }
            version = registry.register_schema("crime", "raw", schema, primary_key="incident_number")
        """
        # Extract schema from DataFrame if needed
        if isinstance(schema, pd.DataFrame):
            schema = self._extract_schema_from_dataframe(schema)

        # Generate version if not provided
        if version is None:
            version = self._generate_version()

        # Create metadata
        metadata = SchemaMetadata(
            dataset=dataset,
            layer=layer,
            version=version,
            created_at=datetime.utcnow(),
            created_by=created_by,
            description=description,
            num_columns=len(schema),
            primary_key=primary_key,
        )

        # Combine schema and metadata
        schema_with_metadata = {
            "metadata": {
                "dataset": metadata.dataset,
                "layer": metadata.layer,
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "created_by": metadata.created_by,
                "description": metadata.description,
                "num_columns": metadata.num_columns,
                "primary_key": metadata.primary_key,
            },
            "schema": schema,
        }

        # Upload versioned schema
        versioned_path = f"{self.schemas_path}/{dataset}/{layer}/{version}.json"
        self._upload_json(versioned_path, schema_with_metadata)

        # Update latest pointer
        latest_path = f"{self.schemas_path}/{dataset}/{layer}/latest.json"
        self._upload_json(latest_path, schema_with_metadata)

        logger.info(
            f"Registered schema for {dataset}/{layer} version {version}",
            extra={"dataset": dataset, "layer": layer, "version": version},
        )

        return version

    def get_schema(
        self,
        dataset: str,
        layer: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        """
        Get schema from registry.

        Args:
            dataset: Dataset name
            layer: Data layer
            version: Specific version (uses latest if not provided)

        Returns:
            Schema dictionary with metadata

        Raises:
            FileNotFoundError: If schema doesn't exist
        """
        if version is None:
            path = f"{self.schemas_path}/{dataset}/{layer}/latest.json"
        else:
            path = f"{self.schemas_path}/{dataset}/{layer}/{version}.json"

        try:
            schema = self._download_json(path)
            logger.debug(
                f"Retrieved schema for {dataset}/{layer}",
                extra={"dataset": dataset, "layer": layer, "version": version},
            )
            return schema
        except Exception as e:
            raise FileNotFoundError(f"Schema not found: {dataset}/{layer} version={version}") from e

    def get_schema_metadata(
        self,
        dataset: str,
        layer: str,
        version: str | None = None,
    ) -> SchemaMetadata:
        """Get schema metadata without full schema."""
        schema = self.get_schema(dataset, layer, version)
        meta = schema["metadata"]
        return SchemaMetadata(
            dataset=meta["dataset"],
            layer=meta["layer"],
            version=meta["version"],
            created_at=datetime.fromisoformat(meta["created_at"]),
            created_by=meta["created_by"],
            description=meta.get("description"),
            num_columns=meta["num_columns"],
            primary_key=meta.get("primary_key"),
        )

    def list_versions(self, dataset: str, layer: str) -> list[str]:
        """
        List all versions for a dataset/layer.

        Returns:
            List of version strings, sorted newest first
        """
        prefix = f"{self.schemas_path}/{dataset}/{layer}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        versions = []
        for blob in blobs:
            filename = Path(blob.name).name
            if filename != "latest.json" and filename.endswith(".json"):
                versions.append(filename.replace(".json", ""))

        # Sort by timestamp (versions are formatted as YYYYMMDD_HHMMSS)
        versions.sort(reverse=True)
        return versions

    def schema_exists(self, dataset: str, layer: str, version: str | None = None) -> bool:
        """Check if a schema exists."""
        try:
            self.get_schema(dataset, layer, version)
            return True
        except FileNotFoundError:
            return False

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        dataset: str,
        layer: str,
        version: str | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate a DataFrame against a registered schema.

        Args:
            df: DataFrame to validate
            dataset: Dataset name
            layer: Data layer
            version: Schema version (uses latest if not provided)

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            is_valid, errors = registry.validate_dataframe(df, "crime", "raw")
            if not is_valid:
                for error in errors:
                    print(f"Validation error: {error}")
        """
        schema_doc = self.get_schema(dataset, layer, version)
        schema = schema_doc["schema"]
        errors = []

        # Check for missing columns
        schema_columns = set(schema.keys())
        df_columns = set(df.columns)

        missing_columns = schema_columns - df_columns
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for extra columns (if strict mode)
        if not self.config.validation.schema.allow_extra_columns:
            extra_columns = df_columns - schema_columns
            if extra_columns:
                errors.append(f"Extra columns not in schema: {extra_columns}")

        # Validate column types
        for col, col_schema in schema.items():
            if col not in df.columns:
                continue

            expected_type = col_schema.get("type")
            nullable = col_schema.get("nullable", True)

            # Check nullability
            if not nullable and df[col].isna().any():
                null_count = df[col].isna().sum()
                errors.append(
                    f"Column '{col}' has {null_count} null values but is marked as non-nullable"
                )

            # Check type compatibility (basic check)
            actual_dtype = str(df[col].dtype)
            if not self._is_type_compatible(actual_dtype, expected_type):
                errors.append(
                    f"Column '{col}' has dtype '{actual_dtype}' but schema expects '{expected_type}'"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _extract_schema_from_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Extract schema from a pandas DataFrame."""
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            nullable = df[col].isna().any()

            # Map pandas dtype to schema type
            if pd.api.types.is_integer_dtype(dtype):
                type_name = "integer"
            elif pd.api.types.is_float_dtype(dtype):
                type_name = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                type_name = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_name = "datetime"
            elif pd.api.types.is_string_dtype(dtype) or dtype is object:
                type_name = "string"
            else:
                type_name = str(dtype)

            schema[col] = {
                "type": type_name,
                "nullable": bool(nullable),
            }

        return schema

    def _is_type_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected type."""
        # Map pandas dtypes to schema types
        type_mappings = {
            "int64": ["integer", "float"],
            "float64": ["float"],
            "object": ["string"],
            "bool": ["boolean"],
            "datetime64[ns]": ["datetime"],
            "string": ["string"],
        }

        compatible_types = type_mappings.get(actual, [actual])
        return expected in compatible_types

    def _generate_version(self) -> str:
        """Generate a version string based on timestamp."""
        return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")

    def _upload_json(self, path: str, data: dict[str, Any]) -> None:
        """Upload JSON data to GCS."""
        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )

    def _download_json(self, path: str) -> dict[str, Any]:
        """Download JSON data from GCS."""
        blob = self.bucket.blob(path)
        content = blob.download_as_string()
        return json.loads(content)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_schema_from_dataframe(
    df: pd.DataFrame,
    primary_key: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """
    Create a schema dictionary from a DataFrame.

    This is a convenience function for creating schemas without
    registering them immediately.

    Args:
        df: Source DataFrame
        primary_key: Primary key column name
        description: Schema description

    Returns:
        Schema dictionary
    """
    registry = SchemaRegistry()
    schema = registry._extract_schema_from_dataframe(df)

    return {
        "description": description,
        "primary_key": primary_key,
        "schema": schema,
    }
