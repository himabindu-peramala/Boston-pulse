"""
Crime Data Ingestion Script
Downloads REAL crime incident reports from Boston Open Data Portal
"""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/crime_ingestion.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def download_crime_data(
    output_path: str = "data/raw/crime_data.csv", year: str = "2023", nrows: int = None
):
    """
    Download REAL crime incident data from Boston Open Data Portal

    Args:
        output_path: Where to save the CSV file
        year: Which year to download (2015-2023)
        nrows: Limit number of rows (None = all data)

    Returns:
        Path to saved file
    """
    try:
        logger.info(f"Starting crime data download for year {year}...")

        # Direct CSV download URLs from Analyze Boston
        crime_urls = {
            "2023": "https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/b973d8cb-eeb2-4e7e-99da-c92938efc9c0/download/tmpvgr7yw3x.csv",
            "2022": "https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/313e56df-6d77-49d2-9c49-ee411f10cf58/download/tmpdfeo3qy2.csv",
            "2021": "https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/f4495ee9-c42c-4019-82c1-d067f07e45d2/download/tmpfap3hfze.csv",
            "2020": "https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/be047094-85fe-4104-a480-4fa3d03f9623/download/tmpkd_w64k_.csv",
            "2019": "https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/34e0ae6b-8c94-4998-ae9e-1b51551fe9ba/download/tmp6w6ts2d7.csv",
        }

        if year not in crime_urls:
            raise ValueError(f"Year {year} not available. Choose from: {list(crime_urls.keys())}")

        url = crime_urls[year]
        logger.info(f"Downloading from: {url}")

        # Use requests with proper headers to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        logger.info("Sending HTTP request with proper headers...")
        response = requests.get(url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()

        logger.info(f"Response received. Status: {response.status_code}")

        # Read CSV from response content
        if nrows:
            logger.info(f"Limiting download to {nrows} rows...")
            df = pd.read_csv(StringIO(response.text), nrows=nrows)
        else:
            logger.info("Downloading full dataset (this may take 1-2 minutes)...")
            df = pd.read_csv(StringIO(response.text))

        logger.info(f"Downloaded {len(df)} crime records from {year}")

        # Basic validation
        if df.empty:
            raise ValueError("Downloaded dataset is empty!")

        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Crime data saved to {output_path}")

        # Log basic statistics
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")

        if "OCCURRED_ON_DATE" in df.columns:
            logger.info(
                f"Date range: {df['OCCURRED_ON_DATE'].min()} to {df['OCCURRED_ON_DATE'].max()}"
            )

        return output_path

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        logger.error("The Boston data portal may be blocking automated downloads.")
        logger.info(
            "Alternative: Download manually from https://data.boston.gov and place in data/raw/"
        )
        raise

    except Exception as e:
        logger.error(f"Error downloading crime data: {str(e)}")
        raise


def validate_crime_data(file_path: str):
    """
    Validate downloaded crime data
    """
    try:
        logger.info(f"Validating crime data from {file_path}...")

        # Try to read the file
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            # Handle completely empty files
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

        # Check for critical issues
        critical_issues = []

        if validation_results["total_rows"] == 0:
            critical_issues.append("Dataset is empty")

        if validation_results["duplicate_rows"] > len(df) * 0.1:
            critical_issues.append(
                f"High duplicate rate: {validation_results['duplicate_rows']} rows"
            )

        # Check for expected columns
        expected_columns = ["INCIDENT_NUMBER", "OFFENSE_CODE", "DISTRICT"]
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            critical_issues.append(f"Missing expected columns: {missing_cols}")

        # Check coordinate validity (if columns exist)
        if "Lat" in df.columns and "Long" in df.columns:
            df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
            df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

            invalid_coords = (
                (df["Lat"] < 42.0)
                | (df["Lat"] > 42.5)
                | (df["Long"] < -71.3)
                | (df["Long"] > -70.9)
            ).sum()

            if invalid_coords > len(df) * 0.2:  # More than 20%
                critical_issues.append(f"High invalid coordinates: {invalid_coords} rows")

        validation_results["critical_issues"] = critical_issues

        if critical_issues:
            logger.warning(f"Validation issues found: {critical_issues}")
        else:
            logger.info("Validation passed: No critical issues found")

        return validation_results

    except Exception as e:
        logger.error(f"Error validating crime data: {str(e)}")
        raise


if __name__ == "__main__":
    # Download crime data from 2023 (limit to 50,000 rows for faster testing)
    # Set nrows=None to download ALL data
    file_path = download_crime_data(year="2023", nrows=None)  # Full dataset

    # Validate the downloaded data
    validation = validate_crime_data(file_path)

    print("\n=== Validation Results ===")
    print(f"Total rows: {validation['total_rows']}")
    print(f"Total columns: {validation['column_count']}")
    print(f"Duplicate rows: {validation['duplicate_rows']}")
    print(f"Total missing values: {validation['missing_values']}")
    print(f"Columns: {validation['columns'][:8]}...")  # Show first 8
    print(f"Critical issues: {validation['critical_issues']}")
