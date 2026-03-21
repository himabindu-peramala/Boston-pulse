"""
Boston Pulse — Data Ingestion Script
Loads all 6 datasets from GCS, chunks them into text,
embeds with HuggingFace, and stores in ChromaDB.

Run once before starting the API:
    python scripts/ingest.py

Re-run anytime pipeline writes new data to GCS.
"""
import logging
import sys
import os

# Add backend root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.gcs_loader import (
    load_crime,
    load_service_311,
    load_food_inspections,
    load_cityscore,
    load_berdo,
    load_street_sweeping,
)
from app.services.embedder import (
    chunk_crime_records,
    chunk_311_records,
    chunk_food_inspection_records,
    chunk_cityscore_records,
    chunk_berdo_records,
    chunk_street_sweeping_records,
)
from app.services.retriever import add_chunks, get_collection_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max rows per dataset to keep ingestion fast
# Increase for production
MAX_ROWS = {
    "crime": 5000,
    "service_311": 5000,
    "food_inspections": 5000,
    "cityscore": 1000,
    "berdo": 1000,
    "street_sweeping": 1000,
}


def ingest_dataset(name, loader_fn, chunker_fn, max_rows):
    """Load, chunk, and ingest one dataset."""
    logger.info(f"Loading {name}...")
    df = loader_fn()

    if df.empty:
        logger.warning(f"{name}: No data found — skipping")
        return 0

    # Sample to keep ingestion manageable
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        logger.info(f"{name}: Sampled {max_rows} rows from {len(df)} total")

    logger.info(f"{name}: Chunking {len(df)} rows...")
    chunks = chunker_fn(df)

    logger.info(f"{name}: Embedding and storing {len(chunks)} chunks...")
    added = add_chunks(chunks)
    logger.info(f"{name}: Done! Added {added} chunks")
    return added


def main():
    logger.info("=" * 50)
    logger.info("Boston Pulse — Data Ingestion Starting")
    logger.info("=" * 50)

    datasets = [
        ("crime",            load_crime,             chunk_crime_records,             MAX_ROWS["crime"]),
        ("service_311",      load_service_311,        chunk_311_records,               MAX_ROWS["service_311"]),
        ("food_inspections", load_food_inspections,   chunk_food_inspection_records,   MAX_ROWS["food_inspections"]),
        ("cityscore",        load_cityscore,          chunk_cityscore_records,         MAX_ROWS["cityscore"]),
        ("berdo",            load_berdo,              chunk_berdo_records,             MAX_ROWS["berdo"]),
        ("street_sweeping",  load_street_sweeping,    chunk_street_sweeping_records,   MAX_ROWS["street_sweeping"]),
    ]

    total_chunks = 0
    ingested_datasets = []

    for name, loader, chunker, max_rows in datasets:
        try:
            count = ingest_dataset(name, loader, chunker, max_rows)
            if count > 0:
                total_chunks += count
                ingested_datasets.append(name)
        except Exception as e:
            logger.error(f"Failed to ingest {name}: {e}")
            continue

    logger.info("=" * 50)
    logger.info(f"Ingestion complete!")
    logger.info(f"Datasets ingested: {ingested_datasets}")
    logger.info(f"Total chunks stored: {total_chunks}")

    stats = get_collection_stats()
    logger.info(f"ChromaDB total chunks: {stats['total_chunks']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
