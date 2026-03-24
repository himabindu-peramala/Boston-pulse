"""
Boston Pulse ML - Publisher Tests.

Tests for ml/models/crime_navigate/publisher.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


class TestPublishScores:
    """Tests for publish_scores function."""

    def test_upserts_to_firestore(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores upserts documents to Firestore."""
        from models.crime_navigate.publisher import publish_scores

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001", "h3_002"],
                "hour_bucket": [0, 1],
                "predicted_danger": [1.5, 2.5],
                "risk_score": [30.0, 70.0],
                "risk_tier": ["LOW", "HIGH"],
            }
        )

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_firestore.return_value = mock_db

        result = publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        assert result.success is True
        assert result.rows_upserted == 2
        mock_batch.commit.assert_called()

    def test_uses_correct_document_key_format(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores uses {h3_index}_{hour_bucket} as document key."""
        from models.crime_navigate.publisher import publish_scores

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [3],
                "predicted_danger": [1.5],
                "risk_score": [50.0],
                "risk_tier": ["MEDIUM"],
            }
        )

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_collection = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_db.collection.return_value = mock_collection
        mock_firestore.return_value = mock_db

        publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        # Check that document was created with correct key
        mock_collection.document.assert_called_with("h3_001_3")

    def test_batches_writes(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores batches writes according to batch_size."""
        from models.crime_navigate.publisher import publish_scores

        # Create more rows than batch_size
        n_rows = 250
        scores_df = pd.DataFrame(
            {
                "h3_index": [f"h3_{i:03d}" for i in range(n_rows)],
                "hour_bucket": [i % 6 for i in range(n_rows)],
                "predicted_danger": np.random.uniform(0, 5, n_rows),
                "risk_score": np.random.uniform(0, 100, n_rows),
                "risk_tier": np.random.choice(["LOW", "MEDIUM", "HIGH"], n_rows),
            }
        )

        sample_cfg["firestore"]["batch_size"] = 100

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_firestore.return_value = mock_db

        result = publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        # Should have committed 3 batches (100 + 100 + 50)
        assert mock_batch.commit.call_count == 3
        assert result.rows_upserted == n_rows

    def test_includes_model_version_in_document(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores includes model_version in each document."""
        from models.crime_navigate.publisher import publish_scores

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "predicted_danger": [1.5],
                "risk_score": [50.0],
                "risk_tier": ["MEDIUM"],
            }
        )

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_firestore.return_value = mock_db

        publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        # Check that batch.set was called with model_version in data
        set_call = mock_batch.set.call_args
        doc_data = set_call[0][1]
        assert doc_data["model_version"] == "20240115"

    def test_records_duration(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores records duration in result."""
        from models.crime_navigate.publisher import publish_scores

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "predicted_danger": [1.5],
                "risk_score": [50.0],
                "risk_tier": ["MEDIUM"],
            }
        )

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_firestore.return_value = mock_db

        result = publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        assert result.duration_seconds >= 0

    def test_uses_custom_collection(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """publish_scores uses collection from config."""
        from models.crime_navigate.publisher import publish_scores

        scores_df = pd.DataFrame(
            {
                "h3_index": ["h3_001"],
                "hour_bucket": [0],
                "predicted_danger": [1.5],
                "risk_score": [50.0],
                "risk_tier": ["MEDIUM"],
            }
        )

        sample_cfg["firestore"]["collection"] = "custom_scores"

        mock_db = MagicMock()
        mock_batch = MagicMock()
        mock_db.batch.return_value = mock_batch
        mock_firestore.return_value = mock_db

        result = publish_scores(scores_df, sample_cfg, "20240115", "2024-01-15")

        mock_db.collection.assert_called_with("custom_scores")
        assert result.firestore_collection == "custom_scores"


class TestVerifyPublish:
    """Tests for verify_publish function."""

    def test_returns_true_when_count_matches(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """verify_publish returns True when document count matches expected."""
        from models.crime_navigate.publisher import verify_publish

        mock_db = MagicMock()
        mock_count_result = MagicMock()
        mock_count_result.get.return_value = [(MagicMock(value=100),)]
        mock_db.collection.return_value.where.return_value.count.return_value = mock_count_result
        mock_firestore.return_value = mock_db

        result = verify_publish(sample_cfg, "20240115", 100)

        assert result is True

    def test_returns_false_when_count_too_low(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """verify_publish returns False when document count is too low."""
        from models.crime_navigate.publisher import verify_publish

        mock_db = MagicMock()
        mock_count_result = MagicMock()
        mock_count_result.get.return_value = [(MagicMock(value=50),)]
        mock_db.collection.return_value.where.return_value.count.return_value = mock_count_result
        mock_firestore.return_value = mock_db

        result = verify_publish(sample_cfg, "20240115", 100)

        assert result is False


class TestGetCollectionStats:
    """Tests for get_collection_stats function."""

    def test_returns_stats(
        self, sample_cfg: dict[str, Any], mock_firestore: Any, mocker: Any
    ) -> None:
        """get_collection_stats returns collection statistics."""
        from models.crime_navigate.publisher import get_collection_stats

        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "h3_index": "h3_001",
            "hour_bucket": 0,
            "risk_score": 50.0,
            "model_version": "20240115",
        }
        mock_db.collection.return_value.limit.return_value.stream.return_value = [mock_doc]
        mock_firestore.return_value = mock_db

        stats = get_collection_stats(sample_cfg)

        assert stats["count"] == 1
        assert "20240115" in stats["model_versions"]
        assert stats["sample"] is not None
