"""
Boston Pulse Backend — GCS Data Loader
Wraps the pipeline's GCSDataIO to load processed Parquet files.
"""
import logging
import sys
import os
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# Correct path to data-pipeline
_PIPELINE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data-pipeline")
)
if _PIPELINE_PATH not in sys.path:
    sys.path.insert(0, _PIPELINE_PATH)

logger.info(f"Pipeline path: {_PIPELINE_PATH}")


def _get_gcs_io():
    try:
        from dags.utils.gcs_io import GCSDataIO
        return GCSDataIO()
    except Exception as e:
        logger.warning(f"GCSDataIO unavailable: {e} — returning empty DataFrames")
        return None


@lru_cache(maxsize=1)
def load_crime() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("crime", "processed")
    except Exception as e:
        logger.warning(f"Could not load crime: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_service_311() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("service_311", "processed")
    except Exception as e:
        logger.warning(f"Could not load 311: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_food_inspections() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("food_inspections", "processed")
    except Exception as e:
        logger.warning(f"Could not load food inspections: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_cityscore() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("cityscore", "processed")
    except Exception as e:
        logger.warning(f"Could not load cityscore: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_berdo() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("berdo", "processed")
    except Exception as e:
        logger.warning(f"Could not load berdo: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_street_sweeping() -> pd.DataFrame:
    gcs = _get_gcs_io()
    if gcs is None:
        return pd.DataFrame()
    try:
        return gcs.read_latest_parquet("street_sweeping", "processed")
    except Exception as e:
        logger.warning(f"Could not load street sweeping: {e}")
        return pd.DataFrame()


def clear_cache():
    load_crime.cache_clear()
    load_service_311.cache_clear()
    load_food_inspections.cache_clear()
    load_cityscore.cache_clear()
    load_berdo.cache_clear()
    load_street_sweeping.cache_clear()
    logger.info("GCS cache cleared")
