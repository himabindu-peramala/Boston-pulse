#!/usr/bin/env python3
"""
Boston Pulse ML - Backfill Baseline Script.

One-off script to create a training baseline for the current production model.
Used for cold-start migration when a production model exists but has no baseline yet.

Usage:
    cd ml
    python -m scripts.backfill_baseline

    # Or directly:
    python scripts/backfill_baseline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add ml/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.crime_navigate.cli import load_config
from models.crime_navigate.feature_loader import load_features
from models.crime_navigate.target_builder import build_targets
from shared.baseline_snapshotter import snapshot_training_baseline
from shared.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Backfill training baseline for current production model."""
    logger.info("Loading config...")
    cfg = load_config()

    logger.info("Initializing registry...")
    registry = ModelRegistry(cfg)

    logger.info("Fetching production model metadata...")
    prod_meta = registry.get_latest_metadata()

    if prod_meta is None:
        logger.error("No production model found in registry. Nothing to backfill.")
        return 1

    version = prod_meta.get("version")
    execution_date = prod_meta.get("execution_date")

    if not version or not execution_date:
        logger.error(f"Invalid production metadata: {prod_meta}")
        return 1

    logger.info(f"Production model: version={version}, execution_date={execution_date}")

    bucket = cfg["data"]["bucket"]

    logger.info("Loading features...")
    features_df, feature_result = load_features(execution_date, cfg, bucket)
    if not feature_result.success:
        logger.error(f"Failed to load features: {feature_result.error}")
        return 1
    logger.info(f"Loaded {feature_result.rows} feature rows")

    logger.info("Building targets...")
    training_df, target_result = build_targets(features_df, execution_date, cfg, bucket)
    if not target_result.success:
        logger.error(f"Failed to build targets: {target_result.error}")
        return 1
    logger.info(f"Built {target_result.rows} training rows")

    logger.info("Snapshotting training baseline...")
    baseline_paths = snapshot_training_baseline(
        training_df=training_df,
        feature_cols=cfg["features"]["input_columns"],
        target_col=cfg["features"]["target_column"],
        version=version,
        cfg=cfg,
    )

    logger.info("Baseline backfill complete!")
    logger.info(f"  Sample URI: {baseline_paths['sample_uri']}")
    logger.info(f"  Stats URI:  {baseline_paths['stats_uri']}")
    logger.info(f"  Latest URI: {baseline_paths['latest_sample_uri']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
