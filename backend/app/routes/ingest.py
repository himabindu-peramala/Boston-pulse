"""
Boston Pulse — Ingest Route
POST /api/ingest
"""
import logging
from fastapi import APIRouter, BackgroundTasks
from app.models.schemas import IngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_ROWS = {
    "crime": 5000,
    "service_311": 5000,
    "food_inspections": 5000,
    "cityscore": 1000,
    "berdo": 1000,
    "street_sweeping": 1000,
}


async def _run_ingestion():
    import os
    from app.core.gcs_loader import clear_cache, load_crime, load_service_311, load_food_inspections, load_cityscore, load_berdo, load_street_sweeping
    from app.services.embedder import chunk_crime_records, chunk_311_records, chunk_food_inspection_records, chunk_cityscore_records, chunk_berdo_records, chunk_street_sweeping_records
    from app.services.retriever import add_chunks

    clear_cache()
    logger.info("Starting re-ingestion...")

    datasets = [
        ("crime",            load_crime,             chunk_crime_records),
        ("service_311",      load_service_311,        chunk_311_records),
        ("food_inspections", load_food_inspections,   chunk_food_inspection_records),
        ("cityscore",        load_cityscore,          chunk_cityscore_records),
        ("berdo",            load_berdo,              chunk_berdo_records),
        ("street_sweeping",  load_street_sweeping,    chunk_street_sweeping_records),
    ]

    for name, loader, chunker in datasets:
        try:
            df = loader()
            if df.empty:
                logger.warning(f"{name}: No data found")
                continue
            max_rows = MAX_ROWS.get(name, 1000)
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42)
            chunks = chunker(df)
            add_chunks(chunks)
            logger.info(f"{name}: Done! Added {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"{name}: Failed — {e}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_ingestion)
    return IngestResponse(
        status="started",
        datasets_ingested=["crime", "service_311", "food_inspections", "cityscore", "berdo", "street_sweeping"],
        total_chunks=0,
        message="Ingestion started in background. ChromaDB will be updated shortly."
    )
