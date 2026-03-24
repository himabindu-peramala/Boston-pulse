"""
Boston Pulse ML - Score Publisher.

Upsert risk scores to Firestore h3_scores collection.
FastAPI loads full collection at startup — zero Firestore reads on the hot path.

Document key format: {h3_index}_{hour_bucket}
Example: "8a2a100d2dfffff_3"
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from google.cloud import firestore

from shared.schemas import PublishResult

logger = logging.getLogger(__name__)


def publish_scores(
    scores_df: pd.DataFrame,
    cfg: dict[str, Any],
    model_version: str,
    execution_date: str,
) -> PublishResult:
    """
    Publish risk scores to Firestore.

    Args:
        scores_df: DataFrame with risk scores
        cfg: parsed crime_navigate_train.yaml
        model_version: version string for the model
        execution_date: "YYYY-MM-DD"

    Returns:
        PublishResult for XCom
    """
    collection = cfg["firestore"]["collection"]
    batch_size = cfg["firestore"]["batch_size"]

    db = firestore.Client()
    start = time.time()
    total = 0

    for i in range(0, len(scores_df), batch_size):
        batch = db.batch()
        chunk = scores_df.iloc[i : i + batch_size]

        for row in chunk.to_dict("records"):
            key = f"{row['h3_index']}_{int(row['hour_bucket'])}"
            doc_ref = db.collection(collection).document(key)

            batch.set(
                doc_ref,
                {
                    "h3_index": row["h3_index"],
                    "hour_bucket": int(row["hour_bucket"]),
                    "risk_score": float(row["risk_score"]),
                    "risk_tier": row["risk_tier"],
                    "predicted_danger": float(row["predicted_danger"]),
                    "model_version": model_version,
                    "updated_at": execution_date,
                },
            )

        batch.commit()
        total += len(chunk)

        if (i + batch_size) % (batch_size * 10) == 0:
            logger.info(f"Published {total:,} / {len(scores_df):,} documents")

    duration = time.time() - start
    logger.info(f"Published {total:,} docs to {collection} in {duration:.1f}s")

    return PublishResult(
        rows_upserted=total,
        firestore_collection=collection,
        duration_seconds=duration,
        model_version=model_version,
        success=True,
    )


def delete_stale_scores(
    cfg: dict[str, Any],
    older_than_version: str,
) -> int:
    """
    Delete scores older than a specific model version.

    Use with caution — this is a destructive operation.
    """
    collection = cfg["firestore"]["collection"]
    db = firestore.Client()

    docs = db.collection(collection).where("model_version", "<", older_than_version).stream()

    deleted = 0
    batch = db.batch()

    for doc in docs:
        batch.delete(doc.reference)
        deleted += 1

        if deleted % 500 == 0:
            batch.commit()
            batch = db.batch()

    if deleted % 500 != 0:
        batch.commit()

    logger.info(f"Deleted {deleted} stale documents")
    return deleted


def get_collection_stats(cfg: dict[str, Any]) -> dict[str, Any]:
    """Get statistics about the Firestore collection."""
    collection = cfg["firestore"]["collection"]
    db = firestore.Client()

    docs = list(db.collection(collection).limit(1000).stream())

    if not docs:
        return {"count": 0, "model_versions": [], "sample": None}

    versions = set()
    for doc in docs:
        data = doc.to_dict()
        if "model_version" in data:
            versions.add(data["model_version"])

    return {
        "count": len(docs),
        "model_versions": sorted(versions),
        "sample": docs[0].to_dict() if docs else None,
    }


def verify_publish(
    cfg: dict[str, Any],
    model_version: str,
    expected_count: int,
) -> bool:
    """Verify that publish was successful."""
    collection = cfg["firestore"]["collection"]
    db = firestore.Client()

    count = (
        db.collection(collection)
        .where("model_version", "==", model_version)
        .count()
        .get()[0][0]
        .value
    )

    if count >= expected_count * 0.99:  # Allow 1% tolerance
        logger.info(f"Publish verified: {count} docs with version {model_version}")
        return True
    else:
        logger.error(f"Publish verification FAILED: expected {expected_count}, found {count}")
        return False
