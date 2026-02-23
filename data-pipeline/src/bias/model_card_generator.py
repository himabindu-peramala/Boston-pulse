"""
Boston Pulse - Model Card Generator

Generate model cards documenting dataset and model characteristics:
- Dataset metadata (size, timeframe, source)
- Schema information
- Data quality metrics
- Fairness evaluation results
- Drift analysis
- Anomaly detection results

Model cards provide transparency and accountability for data pipelines.

Usage:
    generator = ModelCardGenerator(config)

    # Generate a model card
    card = generator.generate_model_card(
        dataset="crime",
        df=crime_df,
        validation_result=validation_result,
        fairness_result=fairness_result,
        description="Boston crime incident reports"
    )

    # Save as Markdown and JSON
    generator.save_model_card(card, format="both")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage

from src.bias.fairness_checker import FairnessResult
from src.shared.config import Settings, get_config
from src.validation.anomaly_detector import AnomalyResult
from src.validation.drift_detector import DriftResult
from src.validation.schema_enforcer import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    """
    Model card for dataset or model documentation.

    Based on model card best practices from Google and others.
    """

    # Basic information
    dataset_name: str
    version: str
    created_at: datetime
    created_by: str
    description: str

    # Dataset characteristics
    row_count: int
    column_count: int
    time_range: dict[str, str] | None = None
    primary_key: str | None = None

    # Data quality
    validation_summary: dict[str, Any] | None = None
    anomaly_summary: dict[str, Any] | None = None
    drift_summary: dict[str, Any] | None = None

    # Fairness
    fairness_summary: dict[str, Any] | None = None
    mitigation_summary: dict[str, Any] | None = None

    # Metadata
    schema_version: str | None = None
    tags: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelCardGenerator:
    """
    Generate model cards for datasets and models.

    Creates both human-readable (Markdown) and machine-readable (JSON)
    documentation of dataset characteristics.
    """

    def __init__(self, config: Settings | None = None):
        """
        Initialize model card generator.

        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or get_config()
        self.bucket_name = self.config.storage.buckets.main
        self.model_cards_path = self.config.storage.paths.model_cards

        # Initialize GCS client
        if self.config.storage.emulator.enabled:
            self.client = storage.Client(
                project="test-project",
                client_options={"api_endpoint": self.config.storage.emulator.host},
            )
        else:
            self.client = storage.Client(project=self.config.gcp_project_id)

        self.bucket = self.client.bucket(self.bucket_name)

    def generate_model_card(
        self,
        dataset: str,
        df: pd.DataFrame,
        description: str,
        validation_result: ValidationResult | None = None,
        fairness_result: FairnessResult | None = None,
        drift_result: DriftResult | None = None,
        anomaly_result: AnomalyResult | None = None,
        mitigation_result: dict | None = None,
        version: str | None = None,
        created_by: str = "system",
        tags: list[str] | None = None,
        primary_key: str | None = None,
    ) -> ModelCard:
        """
        Generate a model card for a dataset.

        Args:
            dataset: Dataset name
            df: Source DataFrame
            description: Dataset description
            validation_result: Optional validation result
            fairness_result: Optional fairness evaluation
            drift_result: Optional drift detection result
            anomaly_result: Optional anomaly detection result
            version: Version string (auto-generated if not provided)
            created_by: Creator identifier
            tags: Optional tags
            primary_key: Primary key column name

        Returns:
            ModelCard object
        """
        logger.info(f"Generating model card for {dataset}")

        # Generate version if not provided
        if version is None:
            version = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")

        # Extract time range if datetime columns exist
        time_range = self._extract_time_range(df)

        # Create model card
        card = ModelCard(
            dataset_name=dataset,
            version=version,
            created_at=datetime.now(UTC),
            created_by=created_by,
            description=description,
            row_count=len(df),
            column_count=len(df.columns),
            time_range=time_range,
            primary_key=primary_key,
            tags=tags or [],
        )

        # Add validation summary
        if validation_result:
            card.validation_summary = self._create_validation_summary(validation_result)

        # Add fairness summary
        if fairness_result:
            card.fairness_summary = self._create_fairness_summary(fairness_result)

        # Add drift summary
        if drift_result:
            card.drift_summary = self._create_drift_summary(drift_result)

        # Add anomaly summary
        if anomaly_result:
            card.anomaly_summary = self._create_anomaly_summary(anomaly_result)

        if mitigation_result:
            card.mitigation_summary = self._create_mitigation_summary(mitigation_result)

        logger.info(f"Generated model card for {dataset} (version {version})")

        return card

    def save_model_card(
        self,
        card: ModelCard,
        format: str = "both",
        local_path: str | None = None,
    ) -> dict[str, str]:
        """
        Save model card to GCS.

        Args:
            card: ModelCard to save
            format: "json", "markdown", or "both"
            local_path: Optional local path (for testing)

        Returns:
            Dictionary with paths where card was saved
        """
        paths = {}

        # Save JSON version
        if format in ("json", "both"):
            json_path = f"{self.model_cards_path}/{card.dataset_name}/{card.version}.json"
            if local_path:
                local_json = Path(local_path) / f"{card.version}.json"
                local_json.parent.mkdir(parents=True, exist_ok=True)
                with open(local_json, "w") as f:
                    json.dump(card.to_dict(), f, indent=2, default=str)
                paths["json"] = str(local_json)
            else:
                self._upload_json(json_path, card.to_dict())
                paths["json"] = f"gs://{self.bucket_name}/{json_path}"

        # Save Markdown version
        if format in ("markdown", "both"):
            markdown_path = f"{self.model_cards_path}/{card.dataset_name}/{card.version}.md"
            markdown_content = self._generate_markdown(card)
            if local_path:
                local_md = Path(local_path) / f"{card.version}.md"
                local_md.parent.mkdir(parents=True, exist_ok=True)
                with open(local_md, "w") as f:
                    f.write(markdown_content)
                paths["markdown"] = str(local_md)
            else:
                self._upload_text(markdown_path, markdown_content)
                paths["markdown"] = f"gs://{self.bucket_name}/{markdown_path}"

        # Update latest pointers
        if not local_path:
            if "json" in paths:
                latest_json = f"{self.model_cards_path}/{card.dataset_name}/latest.json"
                self._upload_json(latest_json, card.to_dict())
            if "markdown" in paths:
                latest_md = f"{self.model_cards_path}/{card.dataset_name}/latest.md"
                self._upload_text(latest_md, markdown_content)

        logger.info(f"Saved model card for {card.dataset_name} to {paths}")

        return paths

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _extract_time_range(self, df: pd.DataFrame) -> dict[str, str] | None:
        """Extract time range from datetime columns."""
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        if not date_cols:
            return None

        # Use first datetime column
        date_col = date_cols[0]
        min_date = df[date_col].min()
        max_date = df[date_col].max()

        return {
            "column": date_col,
            "min": min_date.isoformat() if pd.notna(min_date) else None,
            "max": max_date.isoformat() if pd.notna(max_date) else None,}

    def _create_validation_summary(self, result: Any) -> dict[str, Any]:
        """Create validation summary from ValidationResult object or dict."""
        if isinstance(result, dict):
            return {
                "stage": result.get("stage", "unknown"),
                "is_valid": result.get("is_valid", False),
                "error_count": result.get("error_count", 0),
                "warning_count": result.get("warning_count", 0),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),}

        # Handle object (ValidationResult)
        return {
            "stage": getattr(result, "stage", "unknown"),
            "is_valid": getattr(result, "is_valid", False),
            "error_count": len(getattr(result, "errors", [])),
            "warning_count": len(getattr(result, "warnings", [])),
            "errors": getattr(result, "errors", []),
            "warnings": getattr(result, "warnings", []),}

    def _create_fairness_summary(self, result: Any) -> dict[str, Any]:
        """Create fairness summary from FairnessResult object or dict."""
        if isinstance(result, dict):
            return {
                "slices_evaluated": result.get("slices_evaluated", 0),
                "violation_count": result.get("violation_count", 0),
                "critical_count": result.get("critical_count", 0),
                "warning_count": result.get("warning_count", 0),
                "passes_gate": result.get("passes_fairness_gate", True),
                "critical_violations": result.get("violations", [])[:5],}

        # Handle object (FairnessResult)
        return {
            "slices_evaluated": getattr(result, "slices_evaluated", 0),
            "violation_count": len(getattr(result, "violations", [])),
            "critical_count": len(getattr(result, "critical_violations", [])),
            "warning_count": len(getattr(result, "warning_violations", [])),
            "passes_gate": getattr(result, "passes_fairness_gate", True),
            "critical_violations": [{"metric": v.metric, "message": v.message}
                for v in getattr(result, "critical_violations", [])
            ][:5],
        }

    def _create_drift_summary(self, result: Any) -> dict[str, Any]:
        """Create drift summary from DriftResult object or dict."""
        if isinstance(result, dict):
            return {
                "features_analyzed": result.get("features_analyzed", 0),
                "features_with_drift": len(result.get("drifted_features", [])),
                "warning_count": result.get("warning_count", 0),
                "critical_count": result.get("critical_count", 0),
                "critical_features": result.get("drifted_features", [])[:10],
                "warning_features": result.get("warning_features", [])[:10],}

        # Handle object (DriftResult)
        return {
            "features_analyzed": getattr(result, "features_analyzed", 0),
            "features_with_drift": len(getattr(result, "features_with_drift", [])),
            "warning_count": len(getattr(result, "warning_features", [])),
            "critical_count": len(getattr(result, "critical_features", [])),
            "critical_features": getattr(result, "critical_features", [])[:10],
            "warning_features": getattr(result, "warning_features", [])[:10],}

    def _create_anomaly_summary(self, result: Any) -> dict[str, Any]:
        """Create anomaly summary from AnomalyResult object or dict."""
        if isinstance(result, dict):
            return {
                "total_anomalies": result.get("anomaly_count", 0),
                "critical_count": result.get("critical_count", 0),
                "warning_count": result.get("warning_count", 0),"anomalies_by_type": result.get("anomalies_by_type", {}),
            }

        # Handle object (AnomalyResult)
        return {
            "total_anomalies": len(getattr(result, "anomalies", [])),
            "critical_count": len(getattr(result, "critical_anomalies", [])),
            "warning_count": len(getattr(result, "warning_anomalies", [])),"anomalies_by_type": getattr(result, "anomalies_by_type", {}),
        }

    def _create_mitigation_summary(self, result: dict) -> dict[str, Any]:
        """Create mitigation summary from XCom dict."""
        return {
            "applied": result.get("mitigation_applied", False),
            "strategy": result.get("strategy"),
            "dimension": result.get("dimension"),
            "rows_before": result.get("rows_before"),
            "rows_after": result.get("rows_after"),
            "slices_improved": result.get("slices_improved"),
            "total_slices": result.get("total_slices"),
            "weight_range": result.get("weight_range"),
            "reason": result.get("reason"),  # populated when mitigation skipped
        }

    def _generate_markdown(self, card: ModelCard) -> str:
        """Generate Markdown representation of model card."""
        lines = []

        # Header
        lines.append(f"# Model Card: {card.dataset_name}")
        lines.append("")
        lines.append(f"**Version:** {card.version}  ")
        lines.append(f"**Created:** {card.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
        lines.append(f"**Created By:** {card.created_by}  ")
        lines.append("")

        # Description
        lines.append("## Description")
        lines.append("")
        lines.append(card.description)
        lines.append("")

        # Dataset Characteristics
        lines.append("## Dataset Characteristics")
        lines.append("")
        lines.append(f"- **Rows:** {card.row_count:,}")
        lines.append(f"- **Columns:** {card.column_count}")
        if card.primary_key:
            lines.append(f"- **Primary Key:** {card.primary_key}")
        if card.time_range:
            lines.append(f"- **Time Range:** {card.time_range['min']} to {card.time_range['max']}")
        lines.append("")

        # Tags
        if card.tags:
            lines.append("**Tags:** " + ", ".join(f"`{tag}`" for tag in card.tags))
            lines.append("")

        # Validation Summary
        if card.validation_summary:
            lines.append("## Validation")
            lines.append("")
            vs = card.validation_summary
            lines.append(f"- **Stage:** {vs['stage']}")
            lines.append(f"- **Valid:** {'✓' if vs['is_valid'] else '✗'}")
            lines.append(f"- **Errors:** {vs['error_count']}")
            lines.append(f"- **Warnings:** {vs['warning_count']}")

            if vs["errors"]:
                lines.append("")
                lines.append("**Errors:**")
                for error in vs["errors"]:
                    lines.append(f"- {error}")
            lines.append("")

        # Fairness Summary
        if card.fairness_summary:
            lines.append("## Fairness Evaluation")
            lines.append("")
            fs = card.fairness_summary
            lines.append(f"- **Slices Evaluated:** {fs['slices_evaluated']}")
            lines.append(
                f"- **Violations:** {fs['violation_count']} ({fs['critical_count']} critical)"
            )
            lines.append(f"- **Fairness Gate:** {'PASS ✓' if fs['passes_gate'] else 'FAIL ✗'}")

            if fs["critical_violations"]:
                lines.append("")
                lines.append("**Critical Violations:**")
                for v in fs["critical_violations"]:
                    lines.append(f"- [{v['metric']}] {v['message']}")
            lines.append("")

        if card.mitigation_summary:
            lines.append("## Bias Mitigation")
            lines.append("")
            ms = card.mitigation_summary
            if ms["applied"]:
                lines.append(f"- **Strategy:** {ms['strategy']}")
                lines.append(f"- **Dimension:** {ms['dimension']}")
                lines.append(f"- **Rows:** {ms['rows_before']} → {ms['rows_after']}")
                lines.append(
                    f"- **Slices Improved:** {ms['slices_improved']} / {ms['total_slices']}"
                )
                if ms["weight_range"]:
                    lines.append(f"- **Weight Range:** {ms['weight_range']}")
            else:
                lines.append(f"- **Applied:** No — {ms.get('reason', 'no violations detected')}")
            lines.append("")

        # Drift Summary
        if card.drift_summary:
            lines.append("## Drift Detection")
            lines.append("")
            ds = card.drift_summary
            lines.append(f"- **Features Analyzed:** {ds['features_analyzed']}")
            lines.append(f"- **Features with Drift:** {ds['features_with_drift']}")
            lines.append(f"- **Critical:** {ds['critical_count']}")
            lines.append(f"- **Warning:** {ds['warning_count']}")

            if ds["critical_features"]:
                lines.append("")
                lines.append("**Critical Drift:**")
                for feature in ds["critical_features"]:
                    lines.append(f"- {feature}")
            lines.append("")

        # Anomaly Summary
        if card.anomaly_summary:
            lines.append("## Anomaly Detection")
            lines.append("")
            ans = card.anomaly_summary
            lines.append(f"- **Total Anomalies:** {ans['total_anomalies']}")
            lines.append(f"- **Critical:** {ans['critical_count']}")
            lines.append(f"- **Warning:** {ans['warning_count']}")

            if ans["anomalies_by_type"]:
                lines.append("")
                lines.append("**By Type:**")
                for atype, count in ans["anomalies_by_type"].items():
                    lines.append(f"- {atype}: {count}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*This model card was automatically generated by Boston Pulse Data Pipeline.*")

        return "\n".join(lines)

    def _upload_json(self, path: str, data: dict[str, Any]) -> None:
        """Upload JSON to GCS."""
        blob = self.bucket.blob(path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )

    def _upload_text(self, path: str, content: str) -> None:
        """Upload text to GCS."""
        blob = self.bucket.blob(path)
        blob.upload_from_string(content, content_type="text/markdown")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_model_card(
    dataset: str,
    df: pd.DataFrame,
    description: str,
    config: Settings | None = None,
    **kwargs: Any,
) -> ModelCard:
    """
    Convenience function to create a model card.

    Args:
        dataset: Dataset name
        df: Source DataFrame
        description: Dataset description
        config: Configuration object
        **kwargs: Additional arguments for generate_model_card

    Returns:
        ModelCard object
    """
    generator = ModelCardGenerator(config)
    return generator.generate_model_card(dataset, df, description, **kwargs)
