"""
Boston Pulse — Embedder
...
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
            f"Type: {row.get('offense_description', 'Unknown')}. "
            f"Category: {row.get('offense_category', 'Unknown')}. "
            f"District: {row.get('district', 'Unknown')}. "
            f"Street: {row.get('street', 'Unknown')}. "
            f"Date: {row.get('occurred_on_date', 'Unknown')}. "
            f"Hour: {row.get('hour', 'Unknown')}. "
            f"Day: {row.get('day_of_week', 'Unknown')}. "
            f"Shooting: {row.get('shooting', False)}."
        )
        chunks.append({"text": text, "dataset": "crime", "metadata": row.to_dict()})
    return chunks


def chunk_311_records(df) -> List[dict]:
    """Convert 311 DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston 311 service request. "
            f"Topic: {row.get('case_topic', 'Unknown')}. "
            f"Service: {row.get('service_name', 'Unknown')}. "
            f"Status: {row.get('case_status', 'Unknown')}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Department: {row.get('assigned_department', 'Unknown')}. "
            f"Opened: {row.get('open_date', 'Unknown')}. "
            f"Closed: {row.get('close_date', 'Unknown')}. "
            f"On time: {row.get('on_time', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "service_311", "metadata": row.to_dict()})
    return chunks


def chunk_food_inspection_records(df) -> List[dict]:
    """Convert food inspection DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston food inspection. "
            f"Restaurant: {row.get('businessname', 'Unknown')}. "
            f"Address: {row.get('address', 'Unknown')}. "
            f"Result: {row.get('result', 'Unknown')}. "
            f"Date: {row.get('resultdttm', 'Unknown')}. "
            f"License: {row.get('licenseno', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "food_inspections", "metadata": row.to_dict()})
    return chunks


def chunk_cityscore_records(df) -> List[dict]:
    """Convert CityScore DataFrame rows into text chunks."""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston CityScore metric. "
            f"Metric: {row.get('metric_name', 'Unknown')}. "
            f"Day score: {row.get('day_score', 'Unknown')}. "
            f"Week score: {row.get('week_score', 'Unknown')}. "
            f"Month score: {row.get('month_score', 'Unknown')}. "
            f"Target: {row.get('target', 'Unknown')}. "
            f"Date: {row.get('date', row.get('timestamp', 'Unknown'))}."
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
            f"Address: {row.get('address', 'Unknown')}. "
            f"Type: {row.get('property_type', 'Unknown')}. "
            f"Energy use: {row.get('site_energy_use_kbtu', 'Unknown')} kBtu. "
            f"Emissions: {row.get('total_ghg_emissions', 'Unknown')} metric tons CO2. "
            f"Energy Star score: {row.get('energy_star_score', 'Unknown')}. "
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
            f"Street: {row.get('full_street_name', 'Unknown')}. "
            f"From: {row.get('from_street', 'Unknown')}. "
            f"To: {row.get('to_street', 'Unknown')}. "
            f"District: {row.get('district', 'Unknown')}. "
            f"Side: {row.get('side_of_street', 'Unknown')}. "
            f"Season: {row.get('season_start', 'Unknown')} to {row.get('season_end', 'Unknown')}. "
            f"Frequency: {row.get('week_type', 'Unknown')}. "
            f"Tow zone: {row.get('tow_zone', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "street_sweeping", "metadata": row.to_dict()})
    return chunks