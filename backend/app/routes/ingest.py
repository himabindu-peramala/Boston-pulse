"""
Boston Pulse — Ingest Route
POST /api/ingest
Triggers re-ingestion of all GCS datasets into ChromaDB.
Can be called manually or by Airflow after each DAG run.
"""
import logging
from fastapi import APIRouter, BackgroundTasks
from app.models.schemas import IngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_ingestion():
    """Run the full ingestion pipeline in the background."""
    import os
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    )
    from app.core.gcs_loader import clear_cache, load_crime, load_service_311, load_food_inspections, load_cityscore, load_berdo, load_street_sweeping
    from app.services.embedder import chunk_crime_records, chunk_311_records, chunk_food_inspection_records, chunk_cityscore_records, chunk_berdo_records, chunk_street_sweeping_records
    from app.services.retriever import add_chunks
    import shutil

    clear_cache()
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
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
            chunks = chunker(df)
            add_chunks(chunks)
            logger.info(f"{name}: Done!")
        except Exception as e:
            logger.error(f"{name}: Failed — {e}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(background_tasks: BackgroundTasks):
    """
    Trigger re-ingestion of all GCS datasets into ChromaDB.
    Runs in background so the API doesn't timeout.
    Called by Airflow after each DAG run.
    """
    background_tasks.add_task(_run_ingestion)
    return IngestResponse(
        status="started",
        datasets_ingested=["crime", "service_311", "food_inspections", "cityscore", "berdo", "street_sweeping"],
        total_chunks=0,
        message="Ingestion started in background. ChromaDB will be updated shortly."
    )
