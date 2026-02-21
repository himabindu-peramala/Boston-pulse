"""
Fire Incident Data Ingestion Script
Downloads Boston Fire Incident reports from Analyze Boston
"""

import pandas as pd
import logging
import requests
from io import StringIO
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fire_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_fire_data(
    output_path: str = "data/raw/fire_incidents.csv",
    year: str = "current",
    nrows: int = None
):
    """
    Download Boston Fire Incident data from Analyze Boston

    Args:
        output_path: Where to save the CSV file
        year: 'current', '2013', or '2012'
        nrows: Limit rows (None = all)

    Returns:
        Path to saved file
    """
    try:
        logger.info(f"Starting fire incident data download for year: {year}...")

        fire_urls = {
            "current": "https://data.boston.gov/dataset/ac9e373a-1303-4563-b28e-29070229fdfe/resource/91a38b1f-8439-46df-ba47-a30c48845e06/download/tmp6xl2bsvx.csv",
            "2013": "https://data.boston.gov/dataset/ac9e373a-1303-4563-b28e-29070229fdfe/resource/76771c63-2d95-4095-bf3d-5f22844350d8/download/2013-bostonfireincidentopendata.csv",
            "2012": "https://data.boston.gov/dataset/ac9e373a-1303-4563-b28e-29070229fdfe/resource/64d6aa98-a3aa-4080-a316-b6d493082091/download/2012-bostonfireincidentopendata.csv",
        }

        if year not in fire_urls:
            raise ValueError(f"Year {year} not available. Choose from: {list(fire_urls.keys())}")

        url = fire_urls[year]
        logger.info(f"Downloading from: {url}")

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            ),
            'Accept': 'text/csv',
        }

        response = requests.get(url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()

        if nrows:
            df = pd.read_csv(StringIO(response.text), nrows=nrows)
        else:
            df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise ValueError("Downloaded dataset is empty!")

        logger.info(f"Downloaded {len(df)} fire incident records")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Fire incidents data saved to {output_path}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")

        return output_path

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error downloading fire data: {str(e)}")
        raise


def validate_fire_data(file_path: str):
    """
    Validate the downloaded fire incident data

    Args:
        file_path: Path to raw CSV

    Returns:
        Dictionary of validation results
    """
    try:
        logger.info(f"Validating fire incident data from {file_path}...")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return {
                'total_rows': 0,
                'missing_values': 0,
                'duplicate_rows': 0,
                'columns': [],
                'column_count': 0,
                'critical_issues': ['Dataset is empty']
            }

        validation_results = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'columns': list(df.columns),
            'column_count': len(df.columns)
        }

        critical_issues = []

        if validation_results['total_rows'] == 0:
            critical_issues.append("Dataset is empty")

        # Expected columns
        expected_columns = ['Incident Number', 'Alarm Date', 'Incident Type', 'District']
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            critical_issues.append(f"Missing expected columns: {missing_cols}")

        # Check duplicate incident numbers
        if 'Incident Number' in df.columns:
            dup_incidents = df['Incident Number'].duplicated().sum()
            if dup_incidents > len(df) * 0.1:
                critical_issues.append(f"High duplicate incident rate: {dup_incidents}")

        validation_results['critical_issues'] = critical_issues

        if critical_issues:
            logger.warning(f"Validation issues: {critical_issues}")
        else:
            logger.info("Validation passed: No critical issues found")

        return validation_results

    except Exception as e:
        logger.error(f"Error validating fire data: {str(e)}")
        raise


if __name__ == "__main__":
    file_path = download_fire_data(year="2013", nrows=None)
    validation = validate_fire_data(file_path)

    print("\n=== Validation Results ===")
    print(f"Total rows: {validation['total_rows']}")
    print(f"Total columns: {validation['column_count']}")
    print(f"Duplicate rows: {validation['duplicate_rows']}")
    print(f"Missing values: {validation['missing_values']}")
    print(f"Columns: {validation['columns']}")
    print(f"Critical issues: {validation['critical_issues']}")