"""
Police Stations Data Preprocessing Script
Cleans and transforms Boston Police Station location data
"""

import logging
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/stations_preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Boston city center coordinates
BOSTON_CENTER_LAT = 42.3601
BOSTON_CENTER_LON = -71.0589


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points in kilometers
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def preprocess_stations_data(
    input_path: str = "data/raw/police_stations.csv",
    output_path: str = "data/processed/police_stations_clean.csv",
):
    """
    Preprocess police stations data: clean, normalize, and engineer features

    Args:
        input_path: Path to raw stations CSV
        output_path: Path to save cleaned data

    Returns:
        Path to cleaned file
    """
    try:
        logger.info("Starting police stations preprocessing...")

        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records from {input_path}")
        original_count = len(df)

        # 1. Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_count - len(df)} duplicate rows")

        # 2. Standardize column names
        df.columns = [col.strip().upper() for col in df.columns]

        # 3. Use POINT_X / POINT_Y as canonical lat/lon (more precise than
        # derived)
        if "POINT_X" in df.columns and "POINT_Y" in df.columns:
            df["LON"] = pd.to_numeric(df["POINT_X"], errors="coerce")
            df["LAT"] = pd.to_numeric(df["POINT_Y"], errors="coerce")
        elif "LONGITUDE" in df.columns and "LATITUDE" in df.columns:
            df["LON"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
            df["LAT"] = pd.to_numeric(df["LATITUDE"], errors="coerce")

        logger.info("Standardized coordinate columns to LAT/LON")

        # 4. Validate coordinate bounds (Boston)
        valid_coords = (
            (df["LAT"] >= 42.2) & (df["LAT"] <= 42.5) & (df["LON"] >= -71.2) & (df["LON"] <= -70.9)
        )
        df["VALID_COORDS"] = valid_coords.astype(int)
        invalid_count = (~valid_coords).sum()
        if invalid_count > 0:
            logger.warning(f"{invalid_count} stations have invalid coordinates")

        # 5. Standardize text fields
        for col in ["NAME", "NEIGHBORHOOD", "ADDRESS", "CITY", "DISTRICT"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
                df[col] = df[col].replace("NAN", None)

        logger.info("Standardized text fields")

        # 6. Handle missing values
        if "FT_SQFT" in df.columns:
            df["FT_SQFT"] = pd.to_numeric(df["FT_SQFT"], errors="coerce")
            median_sqft = df["FT_SQFT"].median()
            df["FT_SQFT_MISSING"] = df["FT_SQFT"].isnull().astype(int)
            df["FT_SQFT"] = df["FT_SQFT"].fillna(median_sqft)
            logger.info(f"Filled {
                    df['FT_SQFT_MISSING'].sum()} missing FT_SQFT with median ({
                    median_sqft:.0f})")

        if "STORY_HT" in df.columns:
            df["STORY_HT"] = pd.to_numeric(df["STORY_HT"], errors="coerce")
            df["STORY_HT"] = df["STORY_HT"].fillna(df["STORY_HT"].median())

        # 7. Feature Engineering

        # 7a. Distance from Boston city center (km)
        df["DIST_FROM_CENTER_KM"] = df.apply(
            lambda row: (
                haversine_distance(row["LAT"], row["LON"], BOSTON_CENTER_LAT, BOSTON_CENTER_LON)
                if pd.notnull(row["LAT"]) and pd.notnull(row["LON"])
                else None
            ),
            axis=1,
        )
        logger.info("Computed distance from Boston city center")

        # 7b. Station size category based on square footage
        if "FT_SQFT" in df.columns:
            df["SIZE_CATEGORY"] = pd.cut(
                df["FT_SQFT"],
                bins=[0, 8000, 12000, float("inf")],
                labels=["Small", "Medium", "Large"],
                include_lowest=True,
            )
            logger.info("Created station size categories: Small/Medium/Large")

        # 7c. Is Headquarters flag
        if "NAME" in df.columns:
            df["IS_HEADQUARTERS"] = df["NAME"].str.contains("HEADQUARTERS", na=False).astype(int)

        # 7d. Zone classification (Inner/Outer Boston)
        if "DIST_FROM_CENTER_KM" in df.columns:
            df["ZONE"] = df["DIST_FROM_CENTER_KM"].apply(
                lambda d: (
                    "Inner"
                    if d <= 5
                    else ("Mid" if d <= 10 else "Outer") if pd.notnull(d) else "Unknown"
                )
            )
            logger.info("Classified stations into Inner/Mid/Outer zones")

        # 7e. Normalize building size (z-score) for ML use
        if "FT_SQFT" in df.columns:
            mean_sqft = df["FT_SQFT"].mean()
            std_sqft = df["FT_SQFT"].std()
            df["FT_SQFT_NORMALIZED"] = (df["FT_SQFT"] - mean_sqft) / std_sqft

        # 8. ZIP code cleaning
        if "ZIP" in df.columns:
            df["ZIP"] = df["ZIP"].astype(str).str.strip().str[:5]

        # Save output
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} cleaned records to {output_path}")
        logger.info(f"Final columns: {list(df.columns)}")
        logger.info(
            "New features: LON, LAT, VALID_COORDS, DIST_FROM_CENTER_KM, "
            "SIZE_CATEGORY, IS_HEADQUARTERS, ZONE, FT_SQFT_NORMALIZED"
        )

        return output_path

    except Exception as e:
        logger.error(f"Error preprocessing stations data: {str(e)}")
        raise


def generate_statistics(file_path: str = "data/processed/police_stations_clean.csv"):
    """
    Generate summary statistics about the cleaned stations data

    Args:
        file_path: Path to cleaned CSV

    Returns:
        Dictionary of statistics
    """
    try:
        logger.info("Generating stations statistics...")
        df = pd.read_csv(file_path)

        stats = {
            "total_stations": len(df),
            "columns": list(df.columns),
            "neighborhoods": (
                df["NEIGHBORHOOD"].value_counts().to_dict() if "NEIGHBORHOOD" in df.columns else {}
            ),
            "district_list": df["DISTRICT"].dropna().tolist() if "DISTRICT" in df.columns else [],
            "zone_distribution": (
                df["ZONE"].value_counts().to_dict() if "ZONE" in df.columns else {}
            ),
            "size_distribution": (
                df["SIZE_CATEGORY"].value_counts().to_dict()
                if "SIZE_CATEGORY" in df.columns
                else {}
            ),
            "avg_dist_from_center_km": (
                float(df["DIST_FROM_CENTER_KM"].mean())
                if "DIST_FROM_CENTER_KM" in df.columns
                else None
            ),
            "avg_sqft": float(df["FT_SQFT"].mean()) if "FT_SQFT" in df.columns else None,
            "headquarters_count": (
                int(df["IS_HEADQUARTERS"].sum()) if "IS_HEADQUARTERS" in df.columns else None
            ),
        }

        logger.info("Statistics generated successfully")
        return stats

    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise


if __name__ == "__main__":
    output_file = preprocess_stations_data()
    stats = generate_statistics(output_file)

    print("\n=== Police Stations Statistics ===")
    print(f"Total stations: {stats['total_stations']}")
    print(f"Avg distance from city center: {
            stats['avg_dist_from_center_km']:.2f} km")
    print(f"Avg building size: {stats['avg_sqft']:.0f} sq ft")
    print(f"\nZone distribution: {stats['zone_distribution']}")
    print(f"Size distribution: {stats['size_distribution']}")
    print(f"Neighborhoods: {list(stats['neighborhoods'].keys())}")
