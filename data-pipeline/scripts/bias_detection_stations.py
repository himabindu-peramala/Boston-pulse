import logging
import pandas as pd

logger = logging.getLogger(__name__)


class StationsBiasDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.bias_report = {}

    def detect_geographic_coverage_bias(self):
        if "neighborhood" not in self.df.columns or "station_count" not in self.df.columns:
            logger.warning("Required columns missing for geographic coverage bias detection")
            return self.df

        neighborhood_df = (
            self.df.groupby("neighborhood")["station_count"]
            .sum()
            .reset_index()
        )

        neighborhood_counts = neighborhood_df["neighborhood"].nunique()
        total_stations = neighborhood_df["station_count"].sum()

        if neighborhood_counts == 0:
            logger.warning("No neighborhoods found for bias analysis")
            return self.df

        avg_stations = total_stations / neighborhood_counts

        overrepresented = neighborhood_df[
            neighborhood_df["station_count"] > 1.5 * avg_stations
        ]

        underrepresented = neighborhood_df[
            neighborhood_df["station_count"] < 0.5 * avg_stations
        ]

        self.bias_report["geographic_coverage_bias"] = {
            "neighborhood_distribution": neighborhood_df.to_dict("records"),
            "avg_stations_per_neighborhood": avg_stations,
            "overrepresented": overrepresented["neighborhood"].tolist(),
            "underrepresented": underrepresented["neighborhood"].tolist(),
            "neighborhoods_with_no_station": [],
        }

        logger.info("Geographic coverage bias analysis complete")
        logger.info(
            f"Overrepresented neighborhoods: {overrepresented['neighborhood'].tolist()}"
        )
        logger.info(
            f"Underrepresented neighborhoods: {underrepresented['neighborhood'].tolist()}"
        )

        return neighborhood_df

    def detect_size_equity_bias(self):
        if "FT_SQFT" not in self.df.columns:
            logger.warning("FT_SQFT column not found, skipping size equity analysis")
            return self.df

        avg_size = float(self.df["FT_SQFT"].mean())

        large_stations = self.df[self.df["FT_SQFT"] > 1.5 * avg_size]
        small_stations = self.df[self.df["FT_SQFT"] < 0.5 * avg_size]

        self.bias_report["size_equity_bias"] = {
            "average_station_size": avg_size,
            "large_stations": large_stations.to_dict("records"),
            "small_stations": small_stations.to_dict("records"),
        }

        logger.info("Size equity bias analysis complete")
        logger.info(f"Large stations count: {len(large_stations)}")
        logger.info(f"Small stations count: {len(small_stations)}")

        return self.df

    def run_all_bias_checks(self):
        logger.info("Starting bias detection for police stations")

        self.detect_geographic_coverage_bias()
        self.detect_size_equity_bias()

        logger.info("Bias detection completed successfully")

        return self.bias_report