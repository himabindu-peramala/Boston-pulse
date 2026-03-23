"""
Boston Pulse — Simple GCS Loader
Reads processed Parquet files directly from GCS.
No dependency on data-pipeline code.
"""
import logging
import os
from functools import lru_cache
from io import BytesIO

import pandas as pd
from google.cloud import storage

from app.core.config import settings

logger = logging.getLogger(__name__)

BUCKET_NAME = settings.gcs_bucket_main
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

# Dataset names match GCS folder structure
DATASETS = ["crime", "service_311", "food_inspections", "cityscore", "berdo", "street_sweeping"]


def _get_client():
    """Get authenticated GCS client."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return storage.Client.from_service_account_json(CREDENTIALS_PATH)
    return storage.Client(project="bostonpulse")


def _find_latest_parquet(dataset: str) -> str:
    """Find the latest processed Parquet file for a dataset."""
    client = _get_client()
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"processed/{dataset}/dt="

    blobs = list(bucket.list_blobs(prefix=prefix))
    parquet_blobs = [b for b in blobs if b.name.endswith(".parquet")]

    if not parquet_blobs:
        logger.warning(f"No parquet files found for {dataset} in {BUCKET_NAME}/{prefix}")
        return ""

    # Sort by name (dt=YYYY-MM-DD format sorts chronologically)
    parquet_blobs.sort(key=lambda b: b.name, reverse=True)
    latest = parquet_blobs[0].name
    logger.info(f"{dataset}: latest file = {latest}")
    return latest


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load latest processed Parquet for a dataset from GCS."""
    blob_name = _find_latest_parquet(dataset)
    if not blob_name:
        return pd.DataFrame()

    try:
        client = _get_client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)

        data = blob.download_as_bytes()
        df = pd.read_parquet(BytesIO(data))
        logger.info(f"{dataset}: loaded {len(df)} rows from GCS")
        return df
    except Exception as e:
        logger.error(f"Failed to load {dataset}: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=1)
def load_crime() -> pd.DataFrame:
    return load_dataset("crime")

@lru_cache(maxsize=1)
def load_service_311() -> pd.DataFrame:
    return load_dataset("service_311")

@lru_cache(maxsize=1)
def load_food_inspections() -> pd.DataFrame:
    return load_dataset("food_inspections")

@lru_cache(maxsize=1)
def load_cityscore() -> pd.DataFrame:
    return load_dataset("cityscore")

@lru_cache(maxsize=1)
def load_berdo() -> pd.DataFrame:
    return load_dataset("berdo")

@lru_cache(maxsize=1)
def load_street_sweeping() -> pd.DataFrame:
    return load_dataset("street_sweeping")


def clear_cache():
    for fn in [load_crime, load_service_311, load_food_inspections,
               load_cityscore, load_berdo, load_street_sweeping]:
        fn.cache_clear()
    logger.info("GCS cache cleared")


if __name__ == "__main__":
    """Quick test — load each dataset and print shape."""
    for ds in DATASETS:
        df = load_dataset(ds)
        if not df.empty:
            print(f"{ds}: {df.shape[0]} rows, {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}")
        else:
            print(f"{ds}: NO DATA")