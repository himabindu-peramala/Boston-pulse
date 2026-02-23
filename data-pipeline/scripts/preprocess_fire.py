"""
Fire Incident Data Preprocessing Script
Cleans and transforms Boston Fire Incident data
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/fire_preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Incident type severity mapping based on NFIRS codes
def categorize_incident_severity(incident_type):
    """Categorize fire incident type into severity levels"""
    if pd.isna(incident_type):
        return "Unknown"
    code = int(str(incident_type).strip()) if str(incident_type).strip().isdigit() else 0
    if 100 <= code <= 199:
        return "Fire"  # Actual fires
    elif 200 <= code <= 299:
        return "Explosion"  # Explosions
    elif 300 <= code <= 399:
        return "Rescue"  # Rescue & EMS
    elif 400 <= code <= 499:
        return "Hazmat"  # Hazardous conditions
    elif 500 <= code <= 599:
        return "Service"  # Service calls
    elif 600 <= code <= 699:
        return "Good Intent"  # Good intent calls
    elif 700 <= code <= 799:
        return "False Alarm"  # False alarms
    elif 800 <= code <= 899:
        return "Severe Weather"  # Severe weather
    elif 900 <= code <= 999:
        return "Special"  # Special incidents
    else:
        return "Other"


def preprocess_fire_data(
    input_path: str = "data/raw/fire_incidents.csv",
    output_path: str = "data/processed/fire_incidents_clean.csv",
):
    """
    Preprocess fire incident data: clean, transform, and engineer features

    Args:
        input_path: Path to raw fire incidents CSV
        output_path: Path to save cleaned data

    Returns:
        Path to cleaned file
    """
    try:
        logger.info("Starting fire incident data preprocessing...")

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        original_count = len(df)

        # 1. Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_count - len(df)} duplicate rows")

        # 2. Handle missing values
        missing_before = df.isnull().sum().sum()

        # Drop rows missing critical fields
        if "Incident Number" in df.columns:
            df = df.dropna(subset=["Incident Number"])

        # Fill missing categoricals
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

        # Fill missing numerics with 0
        num_cols = ["Estimated Property Loss", "Estimated Content Loss"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")

        # 3. Parse datetime
        if "Alarm Date" in df.columns and "Alarm Time" in df.columns:
            try:
                df["alarm_datetime"] = pd.to_datetime(
                    df["Alarm Date"].astype(str) + " " + df["Alarm Time"].astype(str),
                    errors="coerce",
                )
                df["year"] = df["alarm_datetime"].dt.year
                df["month"] = df["alarm_datetime"].dt.month
                df["day_of_week"] = df["alarm_datetime"].dt.dayofweek
                df["hour"] = df["alarm_datetime"].dt.hour
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

                df["time_of_day"] = pd.cut(
                    df["hour"],
                    bins=[0, 6, 12, 18, 24],
                    labels=["Night", "Morning", "Afternoon", "Evening"],
                    include_lowest=True,
                )
                logger.info(
                    "Extracted temporal features: year, month, day_of_week, hour, is_weekend, time_of_day"
                )
            except Exception as e:
                logger.warning(f"Could not parse datetime: {e}")

        # 4. Incident severity categorization
        if "Incident Type" in df.columns:
            df["severity_category"] = df["Incident Type"].apply(categorize_incident_severity)
            logger.info("Created severity categories based on NFIRS incident type codes")

        # 5. Financial loss features
        if "Estimated Property Loss" in df.columns and "Estimated Content Loss" in df.columns:
            df["total_loss"] = df["Estimated Property Loss"] + df["Estimated Content Loss"]
            df["has_loss"] = (df["total_loss"] > 0).astype(int)

            # Loss severity buckets
            df["loss_category"] = pd.cut(
                df["total_loss"],
                bins=[-1, 0, 1000, 50000, float("inf")],
                labels=["No Loss", "Minor", "Moderate", "Major"],
            )
            logger.info("Created financial loss features: total_loss, has_loss, loss_category")

        # 6. Standardize text fields
        text_cols = ["District", "Neighborhood", "Incident Description", "Property Use"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()

        logger.info("Standardized text fields")

        # 7. District crime density (incident count per district)
        if "District" in df.columns:
            district_counts = df["District"].value_counts()
            df["district_incident_count"] = df["District"].map(district_counts)
            logger.info("Added district incident count feature")

        # 8. ZIP code cleaning
        if "Zip" in df.columns:
            df["Zip"] = df["Zip"].astype(str).str.strip().str[:5]

        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} cleaned records to {output_path}")
        logger.info(f"Final columns: {list(df.columns)}")
        logger.info(
            "New features: alarm_datetime, year, month, day_of_week, hour, "
            "is_weekend, time_of_day, severity_category, total_loss, has_loss, "
            "loss_category, district_incident_count"
        )

        return output_path

    except Exception as e:
        logger.error(f"Error preprocessing fire data: {str(e)}")
        raise


def generate_statistics(file_path: str = "data/processed/fire_incidents_clean.csv"):
    """
    Generate statistics about the cleaned fire incident data

    Args:
        file_path: Path to cleaned CSV

    Returns:
        Dictionary of statistics
    """
    try:
        logger.info("Generating fire incident statistics...")
        df = pd.read_csv(file_path)

        stats = {
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": (
                    str(df["alarm_datetime"].min()) if "alarm_datetime" in df.columns else None
                ),
                "end": str(df["alarm_datetime"].max()) if "alarm_datetime" in df.columns else None,
            },
            "severity_distribution": (
                df["severity_category"].value_counts().to_dict()
                if "severity_category" in df.columns
                else {}
            ),
            "district_distribution": (
                df["District"].value_counts().to_dict() if "District" in df.columns else {}
            ),
            "loss_category_distribution": (
                df["loss_category"].value_counts().to_dict()
                if "loss_category" in df.columns
                else {}
            ),
            "time_of_day_distribution": (
                df["time_of_day"].value_counts().to_dict() if "time_of_day" in df.columns else {}
            ),
            "total_property_loss": (
                float(df["Estimated Property Loss"].sum())
                if "Estimated Property Loss" in df.columns
                else 0
            ),
            "incidents_with_loss": int(df["has_loss"].sum()) if "has_loss" in df.columns else 0,
        }

        logger.info("Statistics generated successfully")
        return stats

    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise


if __name__ == "__main__":
    output_file = preprocess_fire_data()
    stats = generate_statistics(output_file)

    print("\n=== Fire Incident Statistics ===")
    print(f"Total records: {stats['total_records']}")
    print(f"Date range: {stats['date_range']}")
    print(f"\nSeverity distribution: {stats['severity_distribution']}")
    print(f"\nDistrict distribution: {
            dict(
                list(
                    stats['district_distribution'].items())[
                    :5])}")
    print(f"\nLoss category distribution: {
            stats['loss_category_distribution']}")
    print(f"\nTime of day distribution: {stats['time_of_day_distribution']}")
    print(f"\nTotal property loss: ${stats['total_property_loss']:,.0f}")
    print(f"Incidents with loss: {stats['incidents_with_loss']}")
