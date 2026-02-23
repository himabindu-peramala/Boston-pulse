"""
Crime Data Preprocessing Script
Cleans and transforms crime incident data
"""

import logging
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def preprocess_crime_data(
    input_path: str = "data/raw/crime_data.csv",
    output_path: str = "data/processed/crime_data_clean.csv",
):
    """
    Preprocess crime data: clean, transform, and engineer features

    Args:
        input_path: Path to raw crime data CSV
        output_path: Path to save cleaned data

    Returns:
        Path to cleaned file
    """
    try:
        logger.info("Starting crime data preprocessing...")

        # Load raw data
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")

        # Store original count
        original_count = len(df)

        # 1. Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        # 2. Handle missing values
        missing_before = df.isnull().sum().sum()

        # Drop rows with missing critical fields
        if "INCIDENT_NUMBER" in df.columns:
            df = df.dropna(subset=["INCIDENT_NUMBER"])

        # Fill missing categorical values with 'Unknown'
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")

        # 3. Parse datetime
        if "OCCURRED_ON_DATE" in df.columns:
            df["OCCURRED_ON_DATE"] = pd.to_datetime(df["OCCURRED_ON_DATE"], errors="coerce")

            # Extract temporal features
            df["year"] = df["OCCURRED_ON_DATE"].dt.year
            df["month"] = df["OCCURRED_ON_DATE"].dt.month
            df["day_of_week"] = df["OCCURRED_ON_DATE"].dt.dayofweek  # 0=Monday, 6=Sunday
            df["hour"] = df["OCCURRED_ON_DATE"].dt.hour
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

            # Time of day categories
            df["time_of_day"] = pd.cut(
                df["hour"],
                bins=[0, 6, 12, 18, 24],
                labels=["Night", "Morning", "Afternoon", "Evening"],
                include_lowest=True,
            )

            logger.info(
                "Extracted temporal features: year, month, day_of_week, hour, is_weekend, time_of_day"
            )

        # 4. Validate and clean coordinates
        if "Lat" in df.columns and "Long" in df.columns:
            # Convert to numeric
            df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
            df["Long"] = pd.to_numeric(df["Long"], errors="coerce")

            # Boston coordinate ranges
            boston_lat_range = (42.2, 42.5)
            boston_long_range = (-71.2, -71.0)

            # Mark valid coordinates
            valid_coords = (
                (df["Lat"] >= boston_lat_range[0])
                & (df["Lat"] <= boston_lat_range[1])
                & (df["Long"] >= boston_long_range[0])
                & (df["Long"] <= boston_long_range[1])
            )

            invalid_count = (~valid_coords).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} records with invalid coordinates")
                # Keep them but flag them
                df["valid_coords"] = valid_coords.astype(int)

        # 5. Standardize text fields
        text_cols = ["OFFENSE_DESCRIPTION", "DISTRICT", "STREET"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].str.upper().str.strip()

        logger.info("Standardized text fields")

        # 6. Create offense severity categories (simplified)
        if "OFFENSE_CODE" in df.columns:
            # Convert to numeric
            df["OFFENSE_CODE"] = pd.to_numeric(df["OFFENSE_CODE"], errors="coerce")

            # Simple categorization based on code ranges
            def categorize_severity(code):
                if pd.isna(code):
                    return "Unknown"
                elif code < 1000:
                    return "Serious"  # Major crimes
                elif code < 3000:
                    return "Moderate"
                else:
                    return "Minor"

            df["severity"] = df["OFFENSE_CODE"].apply(categorize_severity)
            logger.info("Created severity categories")

        # 7. Create crime density by district (aggregation)
        if "DISTRICT" in df.columns:
            district_counts = df["DISTRICT"].value_counts()
            df["district_crime_count"] = df["DISTRICT"].map(district_counts)
            logger.info("Added district crime count feature")

        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} cleaned records to {output_path}")

        # Log statistics
        logger.info("Preprocessing complete:")
        logger.info(f"  Original records: {original_count}")
        logger.info(f"  Final records: {len(df)}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(
            "  New features added: year, month, day_of_week, hour, is_weekend, time_of_day, severity, district_crime_count"
        )

        return output_path

    except Exception as e:
        logger.error(f"Error preprocessing crime data: {str(e)}")
        raise


def generate_statistics(file_path: str = "data/processed/crime_data_clean.csv"):
    """
    Generate statistics about the cleaned data

    Args:
        file_path: Path to cleaned data

    Returns:
        Dictionary of statistics
    """
    try:
        logger.info("Generating data statistics...")

        df = pd.read_csv(file_path)

        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df["OCCURRED_ON_DATE"].min())
                if "OCCURRED_ON_DATE" in df.columns
                else None,
                "end": str(df["OCCURRED_ON_DATE"].max())
                if "OCCURRED_ON_DATE" in df.columns
                else None,
            },
            "top_offenses": df["OFFENSE_DESCRIPTION"].value_counts().head(5).to_dict()
            if "OFFENSE_DESCRIPTION" in df.columns
            else {},
            "district_distribution": df["DISTRICT"].value_counts().to_dict()
            if "DISTRICT" in df.columns
            else {},
            "severity_distribution": df["severity"].value_counts().to_dict()
            if "severity" in df.columns
            else {},
            "time_of_day_distribution": df["time_of_day"].value_counts().to_dict()
            if "time_of_day" in df.columns
            else {},
        }

        logger.info("Statistics generated successfully")
        return stats

    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise


if __name__ == "__main__":
    # Run preprocessing
    output_file = preprocess_crime_data()

    # Generate statistics
    stats = generate_statistics(output_file)

    print("\n=== Data Statistics ===")
    print(f"Total records: {stats['total_records']}")
    print(f"Columns: {len(stats['columns'])}")
    print(f"Date range: {stats['date_range']}")

    print("\nTop 5 offenses:")
    for offense, count in list(stats["top_offenses"].items())[:5]:
        print(f"  {offense}: {count}")

    print("\nDistrict distribution:")
    for district, count in list(stats["district_distribution"].items())[:5]:
        print(f"  {district}: {count}")

    print("\nSeverity distribution:")
    for severity, count in stats["severity_distribution"].items():
        print(f"  {severity}: {count}")

    print("\nTime of day distribution:")
    for time, count in stats["time_of_day_distribution"].items():
        print(f"  {time}: {count}")
