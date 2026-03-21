"""
Boston Pulse — Embedder
Converts text chunks into vectors using HuggingFace sentence-transformers.
Free, local, no API key needed. Professor approved!
"""
import logging
from typing import List
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded!")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors.
    
    Args:
        texts: List of text chunks to embed
    
    Returns:
        List of embedding vectors
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for similarity search.
    
    Args:
        query: User's question
    
    Returns:
        Embedding vector
    """
    model = _get_model()
    embedding = model.encode([query])
    return embedding[0].tolist()


# =============================================================================
# Text chunking helpers — converts DataFrame rows into text chunks
# =============================================================================

def chunk_crime_records(df) -> List[dict]:
    """Convert crime DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Crime incident in Boston. "
            f"Type: {row.get('offense_description', row.get('OFFENSE_DESCRIPTION', 'Unknown'))}. "
            f"Neighborhood: {row.get('neighborhood', row.get('DISTRICT', 'Unknown'))}. "
            f"Date: {row.get('occurred_on_date', row.get('OCCURRED_ON_DATE', 'Unknown'))}. "
            f"Street: {row.get('street', row.get('STREET', 'Unknown'))}."
        )
        chunks.append({"text": text, "dataset": "crime", "metadata": row.to_dict()})
    return chunks


def chunk_311_records(df) -> List[dict]:
    """Convert 311 DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston 311 service request. "
            f"Type: {row.get('type', row.get('reason', 'Unknown'))}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Status: {row.get('status', 'Unknown')}. "
            f"Opened: {row.get('open_dt', 'Unknown')}. "
            f"Resolution time: {row.get('time_to_resolution', 'Unknown')} days."
        )
        chunks.append({"text": text, "dataset": "service_311", "metadata": row.to_dict()})
    return chunks


def chunk_food_inspection_records(df) -> List[dict]:
    """Convert food inspection DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston food inspection. "
            f"Restaurant: {row.get('businessname', row.get('dbaname', 'Unknown'))}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Result: {row.get('result', 'Unknown')}. "
            f"Violations: {row.get('violdesc', row.get('violation_description', 'None'))}. "
            f"Date: {row.get('inspectiondate', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "food_inspections", "metadata": row.to_dict()})
    return chunks


def chunk_cityscore_records(df) -> List[dict]:
    """Convert CityScore DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston CityScore metric. "
            f"Metric: {row.get('score_name', 'Unknown')}. "
            f"Score: {row.get('score_normalized', row.get('day_score', 'Unknown'))}. "
            f"Date: {row.get('score_dt', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "cityscore", "metadata": row.to_dict()})
    return chunks


def chunk_berdo_records(df) -> List[dict]:
    """Convert BERDO DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston building emissions (BERDO). "
            f"Property: {row.get('property_name', 'Unknown')}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Energy use: {row.get('site_eui', 'Unknown')} kBtu/sqft. "
            f"Emissions: {row.get('total_ghg_emissions', 'Unknown')} metric tons CO2. "
            f"Year: {row.get('reporting_year', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "berdo", "metadata": row.to_dict()})
    return chunks


def chunk_street_sweeping_records(df) -> List[dict]:
    """Convert street sweeping DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston street sweeping schedule. "
            f"Street: {row.get('street_name', row.get('full_street', 'Unknown'))}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Schedule: {row.get('day_of_week', 'Unknown')} "
            f"{row.get('start_time', '')} - {row.get('end_time', '')}. "
            f"District: {row.get('district', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "street_sweeping", "metadata": row.to_dict()})
    return chunks
