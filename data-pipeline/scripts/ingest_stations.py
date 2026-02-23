"""
Police Stations Data Ingestion Script
Downloads Boston Police Station locations from Boston Open Data Portal (GeoJSON)
"""
import os

os.makedirs("logs", exist_ok=True)

import logging
from pathlib import Path

import pandas as pd
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/stations_ingestion.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def download_stations_data(output_path: str = "data/raw/police_stations.csv"):
    """
    Download Boston Police Station data from Analyze Boston (GeoJSON → CSV)

    Args:
        output_path: Where to save the CSV file

    Returns:
        Path to saved file
    """
    try:
        logger.info("Starting police stations data download...")

        url = (
            "https://data.boston.gov/dataset/8aa8671a-94fb-4bdd-9283-4a25d7d640cc"
            "/resource/223512fc-64b1-40ae-83dc-b74cba68f18c"
            "/download/boston_police_stations_bpd_only.geojson"
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        logger.info(f"Downloading from: {url}")
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        geojson = response.json()
        features = geojson.get("features", [])

        if not features:
            raise ValueError("GeoJSON has no features!")

        # Flatten GeoJSON features into rows
        records = []
        for feature in features:
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None])
            props["longitude"] = coords[0]
            props["latitude"] = coords[1]
            records.append(props)

        df = pd.DataFrame(records)
        logger.info(f"Downloaded {len(df)} police station records")

        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Police stations data saved to {output_path}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")

        return output_path

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        raise

    except Exception as e:
        logger.error(f"Error downloading police stations data: {str(e)}")
        raise


def validate_stations_data(file_path: str):
    """
    Validate the downloaded police stations data

    Args:
        file_path: Path to raw CSV

    Returns:
        Dictionary of validation results
    """
    try:
        logger.info(f"Validating police stations data from {file_path}...")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return {
                "total_rows": 0,
                "missing_values": 0,
                "duplicate_rows": 0,
                "columns": [],
                "column_count": 0,
                "critical_issues": ["Dataset is empty"],
            }

        validation_results = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "columns": list(df.columns),
            "column_count": len(df.columns),
        }

        critical_issues = []

        if validation_results["total_rows"] == 0:
            critical_issues.append("Dataset is empty")

        # Expected columns based on BPD GeoJSON schema
        expected_columns = ["NAME", "District", "NEIGHBORHOOD", "ADDRESS"]
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            critical_issues.append(f"Missing expected columns: {missing_cols}")

        # Validate coordinate bounds for Boston
        for lat_col in ["latitude", "POINT_Y"]:
            if lat_col in df.columns:
                df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                invalid = ((df[lat_col] < 42.2) | (df[lat_col] > 42.5)).sum()
                if invalid > 0:
                    critical_issues.append(f"{invalid} stations have invalid latitude values")
                break

        # Check for duplicate station names
        if "NAME" in df.columns:
            dup_names = df["NAME"].duplicated().sum()
            if dup_names > 0:
                critical_issues.append(f"{dup_names} duplicate station names found")

        validation_results["critical_issues"] = critical_issues

        if critical_issues:
            logger.warning(f"Validation issues: {critical_issues}")
        else:
            logger.info("Validation passed: No critical issues found")

        return validation_results

    except Exception as e:
        logger.error(f"Error validating stations data: {str(e)}")
        raise


if __name__ == "__main__":
    file_path = download_stations_data()

    validation = validate_stations_data(file_path)

    print("\n=== Validation Results ===")
    print(f"Total rows: {validation['total_rows']}")
    print(f"Total columns: {validation['column_count']}")
    print(f"Duplicate rows: {validation['duplicate_rows']}")
    print(f"Missing values: {validation['missing_values']}")
    print(f"Columns: {validation['columns']}")
    print(f"Critical issues: {validation['critical_issues']}")
