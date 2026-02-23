import logging
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_stations_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing for police stations")

    if df.empty:
        logger.warning("Input dataframe is empty")
        return df

    # Example fill
    if "FT_SQFT" in df.columns:
        missing_count = df["FT_SQFT"].isna().sum()

        df["FT_SQFT"] = df["FT_SQFT"].fillna(df["FT_SQFT"].median())

        logger.info(f"Filled {missing_count} missing FT_SQFT values")

    logger.info("Preprocessing complete")
    return df


def generate_statistics(df: pd.DataFrame) -> dict:
    logger.info("Generating station statistics")

    stats = {
        "total_records": len(df),
        "column_count": len(df.columns),
    }

    logger.info(f"Generated statistics: {stats}")

    return stats
