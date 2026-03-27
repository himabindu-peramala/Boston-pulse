"""
Boston Pulse Backend — GCS Data Loader
Reads processed Parquet files directly from GCS.
"""
import logging
import os
from functools import lru_cache
from io import BytesIO

import pandas as pd

logger = logging.getLogger(__name__)


def _get_bucket():
    from google.cloud import storage
    bucket_name = os.getenv("GCS_BUCKET_MAIN", "boston-pulse-data-prod")
    client = storage.Client()
    return client.bucket(bucket_name)


def _load_latest(dataset: str) -> pd.DataFrame:
    """Load latest parquet file for a dataset directly from GCS."""
    import json
    try:
        bucket = _get_bucket()
        # Try latest.json pointer first
        pointer_blob = bucket.blob(f"processed/{dataset}/latest.json")
        if pointer_blob.exists():
            pointer = json.loads(pointer_blob.download_as_string())
            execution_date = pointer["execution_date"]
            path = f"processed/{dataset}/dt={execution_date}/data.parquet"
            blob = bucket.blob(path)
            content = blob.download_as_bytes()
            df = pd.read_parquet(BytesIO(content))
            logger.info(f"Loaded {dataset}: {len(df)} rows from {path}")
            return df
    except Exception as e:
        logger.warning(f"Could not load {dataset}: {e}")
    return pd.DataFrame()


@lru_cache(maxsize=1)
def load_crime() -> pd.DataFrame:
    return _load_latest("crime")

@lru_cache(maxsize=1)
def load_service_311() -> pd.DataFrame:
    return _load_latest("service_311")

@lru_cache(maxsize=1)
def load_food_inspections() -> pd.DataFrame:
    return _load_latest("food_inspections")

@lru_cache(maxsize=1)
def load_cityscore() -> pd.DataFrame:
    return _load_latest("cityscore")

@lru_cache(maxsize=1)
def load_berdo() -> pd.DataFrame:
    return _load_latest("berdo")

@lru_cache(maxsize=1)
def load_street_sweeping() -> pd.DataFrame:
    return _load_latest("street_sweeping")

def clear_cache():
    load_crime.cache_clear()
    load_service_311.cache_clear()
    load_food_inspections.cache_clear()
    load_cityscore.cache_clear()
    load_berdo.cache_clear()
    load_street_sweeping.cache_clear()
    logger.info("GCS cache cleared")
