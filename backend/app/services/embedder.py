"""
Boston Pulse — Embedder
Converts text chunks into vectors using HuggingFace sentence-transformers.
"""
import logging
from typing import List
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
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    model = _get_model()
    embedding = model.encode([query])
    return embedding[0].tolist()


def chunk_crime_records(df) -> List[dict]:
    """Crime columns: incident_number, offense_category, offense_description,
    district, occurred_on_date, street, lat, long"""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Crime incident in Boston. "
            f"Type: {row.get('offense_description', 'Unknown')}. "
            f"Category: {row.get('offense_category', 'Unknown')}. "
            f"District: {row.get('district', 'Unknown')}. "
            f"Date: {row.get('occurred_on_date', 'Unknown')}. "
            f"Street: {row.get('street', 'Unknown')}. "
            f"Day: {row.get('day_of_week', 'Unknown')}. "
            f"Hour: {row.get('hour', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "crime", "metadata": {}})
    return chunks


def chunk_311_records(df) -> List[dict]:
    """311 columns: case_id, open_date, close_date, case_topic, service_name,
    assigned_department, case_status, neighborhood, on_time"""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston 311 service request. "
            f"Service: {row.get('service_name', 'Unknown')}. "
            f"Topic: {row.get('case_topic', 'Unknown')}. "
            f"Neighborhood: {row.get('neighborhood', 'Unknown')}. "
            f"Status: {row.get('case_status', 'Unknown')}. "
            f"Opened: {row.get('open_date', 'Unknown')}. "
            f"Closed: {row.get('close_date', 'Unknown')}. "
            f"On time: {row.get('on_time', 'Unknown')}. "
            f"Department: {row.get('assigned_department', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "service_311", "metadata": {}})
    return chunks


def chunk_food_inspection_records(df) -> List[dict]:
    """Food columns: businessname, licenseno, result, resultdttm, address, zip"""
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston food inspection. "
            f"Restaurant: {row.get('businessname', 'Unknown')}. "
            f"Result: {row.get('result', 'Unknown')}. "
            f"Date: {row.get('resultdttm', 'Unknown')}. "
            f"Address: {row.get('address', 'Unknown')}. "
            f"Zip: {row.get('zip', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "food_inspections", "metadata": {}})
    return chunks


def chunk_cityscore_records(df) -> List[dict]:
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston CityScore metric. "
            f"Metric: {row.get('score_name', 'Unknown')}. "
            f"Score: {row.get('score_normalized', row.get('day_score', 'Unknown'))}. "
            f"Date: {row.get('score_dt', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "cityscore", "metadata": {}})
    return chunks


def chunk_berdo_records(df) -> List[dict]:
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
        chunks.append({"text": text, "dataset": "berdo", "metadata": {}})
    return chunks


def chunk_street_sweeping_records(df) -> List[dict]:
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"Boston street sweeping schedule. "
            f"Street: {row.get('street_name', row.get('streetname', 'Unknown'))}. "
            f"Neighborhood: {row.get('neighborhood', row.get('ward', 'Unknown'))}. "
            f"Schedule: {row.get('day_of_week', 'Unknown')} "
            f"{row.get('start_time', '')} - {row.get('end_time', '')}. "
            f"District: {row.get('district', 'Unknown')}."
        )
        chunks.append({"text": text, "dataset": "street_sweeping", "metadata": {}})
    return chunks
