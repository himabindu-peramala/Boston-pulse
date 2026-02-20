"""
Boston Pulse - Crime Dataset

Reference implementation for crime incident data from Boston PD.

Components:
    - CrimeIngester: Fetches crime data from Analyze Boston API
    - CrimePreprocessor: Cleans and validates crime data
    - CrimeFeatureBuilder: Creates crime-related features

Data Source:
    Boston Police Department Crime Incident Reports
    https://data.boston.gov/dataset/crime-incident-reports

Usage:
    from src.datasets.crime import CrimeIngester, CrimePreprocessor, CrimeFeatureBuilder

    # Ingest
    ingester = CrimeIngester()
    result = ingester.run(execution_date="2024-01-15")
    raw_df = ingester.get_data()

    # Preprocess
    preprocessor = CrimePreprocessor()
    result = preprocessor.run(raw_df, execution_date="2024-01-15")
    processed_df = preprocessor.get_data()

    # Build features
    builder = CrimeFeatureBuilder()
    result = builder.run(processed_df, execution_date="2024-01-15")
    features_df = builder.get_data()
"""

from src.datasets.crime.features import CrimeFeatureBuilder, build_crime_features
from src.datasets.crime.ingest import CrimeIngester, get_crime_sample, ingest_crime_data
from src.datasets.crime.preprocess import CrimePreprocessor, preprocess_crime_data

__all__ = [
    "CrimeIngester",
    "CrimePreprocessor",
    "CrimeFeatureBuilder",
    "ingest_crime_data",
    "preprocess_crime_data",
    "build_crime_features",
    "get_crime_sample",
]
