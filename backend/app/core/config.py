"""
Boston Pulse — Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):

    gcp_project_id: Optional[str] = None
    gcs_bucket_main: str = "boston-pulse-data"
    gcs_path_processed: str = "processed"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_tokens: int = 1024

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "boston_pulse"

    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    top_k_results: int = 5
    max_history_turns: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
