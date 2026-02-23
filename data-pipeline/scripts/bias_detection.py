"""
Bias Detection and Fairness Analysis
Analyzes crime data for bias across different demographic and geographic slices
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/bias_detection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BiasDetector:
    """Detect and analyze bias in crime data"""

    def __init__(self, data_path: str = "data/processed/crime_data_clean.csv"):
        """Initialize bias detector with processed crime data"""
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} crime records")

        self.bias_report = {
            "geographic_bias": {},
            "temporal_bias": {},
            "severity_bias": {},
            "mitigation_recommendations": [],
        }

    def detect_geographic_bias(self):
        """Analyze bias across districts and neighborhoods"""
        logger.info("Analyzing geographic bias...")

        if "DISTRICT" not in self.df.columns:
            logger.warning("DISTRICT column not found, skipping geographic bias analysis")
            return

        # Calculate crime density by district
        district_counts = self.df["DISTRICT"].value_counts()
        total_crimes = len(self.df)

        district_stats = []
        for district, count in district_counts.items():
            percentage = (count / total_crimes) * 100
            district_stats.append(
                {"district": district, "crime_count": count, "percentage": percentage}
            )

        district_df = pd.DataFrame(district_stats).sort_values("crime_count", ascending=False)

        # Detect over-representation (districts with >2x average)
        avg_crimes = total_crimes / len(district_counts)
        overrepresented = district_df[district_df["crime_count"] > 2 * avg_crimes]
        underrepresented = district_df[district_df["crime_count"] < 0.5 * avg_crimes]

        self.bias_report["geographic_bias"] = {
            "district_distribution": district_df.to_dict("records"),
            "average_crimes_per_district": avg_crimes,
            "overrepresented_districts": overrepresented["district"].tolist(),
            "underrepresented_districts": underrepresented["district"].tolist(),
        }

        logger.info("Geographic bias analysis complete:")
        logger.info(f"  Overrepresented districts: {overrepresented['district'].tolist()}")
        logger.info(f"  Underrepresented districts: {underrepresented['district'].tolist()}")

        return district_df

    def detect_temporal_bias(self):
        """Analyze bias across time periods"""
        logger.info("Analyzing temporal bias...")

        if "time_of_day" not in self.df.columns:
            logger.warning("time_of_day column not found, skipping temporal bias")
            return

        # Analyze crime distribution by time of day
        time_distribution = self.df["time_of_day"].value_counts()
        total_crimes = len(self.df)

        time_stats = []
        for time_period, count in time_distribution.items():
            percentage = (count / total_crimes) * 100
            time_stats.append(
                {"time_period": time_period, "crime_count": count, "percentage": percentage}
            )

        time_df = pd.DataFrame(time_stats).sort_values("crime_count", ascending=False)

        # Expected: roughly 25% per time period (4 periods)
        expected_percentage = 25.0

        # Detect bias (>10% deviation from expected)
        time_df["deviation"] = abs(time_df["percentage"] - expected_percentage)
        biased_periods = time_df[time_df["deviation"] > 10]

        self.bias_report["temporal_bias"] = {
            "time_distribution": time_df.to_dict("records"),
            "biased_periods": biased_periods["time_period"].tolist(),
            "expected_percentage": expected_percentage,
        }

        logger.info("Temporal bias analysis complete:")
        logger.info(f"  Biased time periods: {biased_periods['time_period'].tolist()}")

        return time_df

    def detect_severity_bias(self):
        """Analyze bias in crime severity categorization"""
        logger.info("Analyzing severity bias...")

        if "severity" not in self.df.columns:
            logger.warning("severity column not found, skipping severity bias")
            return

        # Overall severity distribution
        severity_distribution = self.df["severity"].value_counts()
        total_crimes = len(self.df)

        severity_stats = {"overall_distribution": {}}

        for severity, count in severity_distribution.items():
            percentage = (count / total_crimes) * 100
            severity_stats["overall_distribution"][severity] = {
                "count": count,
                "percentage": percentage,
            }

        # Severity distribution by district (check if certain districts have different patterns)
        if "DISTRICT" in self.df.columns:
            district_severity = (
                pd.crosstab(self.df["DISTRICT"], self.df["severity"], normalize="index") * 100
            )

            severity_stats["by_district"] = district_severity.to_dict()

            # Detect districts with unusual severity patterns
            # Calculate deviation from overall severity distribution
            overall_serious_pct = (
                severity_stats["overall_distribution"].get("Serious", {}).get("percentage", 0)
            )

            biased_districts = []
            for district in district_severity.index:
                if "Serious" in district_severity.columns:
                    district_serious_pct = district_severity.loc[district, "Serious"]
                    deviation = abs(district_serious_pct - overall_serious_pct)

                    if deviation > 15:  # More than 15% deviation
                        biased_districts.append(
                            {
                                "district": district,
                                "serious_crime_pct": district_serious_pct,
                                "deviation": deviation,
                            }
                        )

            severity_stats["biased_districts"] = biased_districts

            if biased_districts:
                logger.warning(
                    f"Found {len(biased_districts)} districts with unusual severity patterns"
                )

        self.bias_report["severity_bias"] = severity_stats

        return severity_stats

    def detect_coordinate_bias(self):
        """Analyze bias in geographic coordinate reporting"""
        logger.info("Analyzing coordinate reporting bias...")

        if "valid_coords" not in self.df.columns:
            logger.warning("valid_coords column not found, skipping coordinate bias")
            return

        # Check if certain districts have more missing/invalid coordinates
        if "DISTRICT" in self.df.columns:
            coord_quality = self.df.groupby("DISTRICT")["valid_coords"].agg(
                [("total", "count"), ("valid", "sum"), ("invalid", lambda x: (x == 0).sum())]
            )

            coord_quality["invalid_percentage"] = (
                coord_quality["invalid"] / coord_quality["total"]
            ) * 100

            # Detect districts with high invalid coordinate rates (>10%)
            problematic_districts = coord_quality[coord_quality["invalid_percentage"] > 10]

            logger.info("Coordinate quality analysis:")
            logger.info(f"  Districts with >10% invalid coordinates: {len(problematic_districts)}")

            if len(problematic_districts) > 0:
                logger.warning(f"Problematic districts: {problematic_districts.index.tolist()}")
                self.bias_report["mitigation_recommendations"].append(
                    "Improve coordinate collection in districts with high invalid rates"
                )

    def generate_bias_report(self, output_path: str = "data/processed/bias_report.txt"):
        """Generate comprehensive bias detection report"""
        logger.info("Generating bias detection report...")

        # Run all bias detection analyses
        self.detect_geographic_bias()
        self.detect_temporal_bias()
        self.detect_severity_bias()
        self.detect_coordinate_bias()

        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("BIAS DETECTION REPORT - BOSTON CRIME DATA\n")
            f.write("=" * 80 + "\n\n")

            # Geographic Bias
            f.write("1. GEOGRAPHIC BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")

            geo_bias = self.bias_report.get("geographic_bias", {})
            if geo_bias:
                f.write(
                    f"Average crimes per district: {geo_bias.get('average_crimes_per_district', 0):.0f}\n\n"
                )

                f.write("Overrepresented Districts (>2x average):\n")
                for district in geo_bias.get("overrepresented_districts", []):
                    f.write(f"  - {district}\n")

                f.write("\nUnderrepresented Districts (<0.5x average):\n")
                for district in geo_bias.get("underrepresented_districts", []):
                    f.write(f"  - {district}\n")

                f.write("\nTop 5 Districts by Crime Count:\n")
                for item in geo_bias.get("district_distribution", [])[:5]:
                    f.write(
                        f"  {item['district']}: {item['crime_count']} ({item['percentage']:.2f}%)\n"
                    )

            f.write("\n")

            # Temporal Bias
            f.write("2. TEMPORAL BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")

            temp_bias = self.bias_report.get("temporal_bias", {})
            if temp_bias:
                f.write(
                    f"Expected distribution per time period: {temp_bias.get('expected_percentage', 25):.1f}%\n\n"
                )

                f.write("Crime Distribution by Time of Day:\n")
                for item in temp_bias.get("time_distribution", []):
                    deviation = item.get("deviation", 0)
                    status = "⚠️ BIASED" if deviation > 10 else "✓ Normal"
                    f.write(
                        f"  {item['time_period']}: {item['crime_count']} ({item['percentage']:.2f}%) [{status}]\n"
                    )

            f.write("\n")

            # Severity Bias
            f.write("3. SEVERITY BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")

            sev_bias = self.bias_report.get("severity_bias", {})
            if sev_bias:
                f.write("Overall Severity Distribution:\n")
                for severity, stats in sev_bias.get("overall_distribution", {}).items():
                    f.write(f"  {severity}: {stats['count']} ({stats['percentage']:.2f}%)\n")

                if sev_bias.get("biased_districts"):
                    f.write("\nDistricts with Unusual Severity Patterns:\n")
                    for item in sev_bias["biased_districts"]:
                        f.write(
                            f"  {item['district']}: {item['serious_crime_pct']:.1f}% serious crimes (deviation: {item['deviation']:.1f}%)\n"
                        )

            f.write("\n")

            # Mitigation Recommendations
            f.write("4. BIAS MITIGATION RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            recommendations = self.bias_report.get("mitigation_recommendations", [])

            # Add standard recommendations
            if geo_bias.get("overrepresented_districts"):
                recommendations.append(
                    "Geographic Bias: Normalize crime scores by population density to avoid unfairly flagging high-density districts"
                )

            if temp_bias.get("biased_periods"):
                recommendations.append(
                    "Temporal Bias: Weight time-of-day predictions to account for reporting bias (e.g., nighttime crimes may be under-reported)"
                )

            if not recommendations:
                recommendations.append("No significant bias detected requiring mitigation")

            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF BIAS DETECTION REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Bias detection report saved to {output_path}")

        return self.bias_report

    def visualize_bias(self, output_dir: str = "data/processed/visualizations"):
        """Create visualizations of detected bias"""
        logger.info("Creating bias visualizations...")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Plot 1: Crime distribution by district
        if "DISTRICT" in self.df.columns:
            plt.figure(figsize=(12, 6))
            district_counts = self.df["DISTRICT"].value_counts().head(10)
            district_counts.plot(kind="bar", color="steelblue")
            plt.title("Crime Distribution by District (Top 10)", fontsize=14, fontweight="bold")
            plt.xlabel("District")
            plt.ylabel("Number of Crimes")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/geographic_bias.png", dpi=300)
            plt.close()
            logger.info("Saved geographic bias visualization")

        # Plot 2: Crime distribution by time of day
        if "time_of_day" in self.df.columns:
            plt.figure(figsize=(10, 6))
            time_counts = self.df["time_of_day"].value_counts()
            colors = [
                "#FF6B6B" if count > time_counts.mean() * 1.2 else "#4ECDC4"
                for count in time_counts.values
            ]
            time_counts.plot(kind="bar", color=colors)
            plt.title("Crime Distribution by Time of Day", fontsize=14, fontweight="bold")
            plt.xlabel("Time Period")
            plt.ylabel("Number of Crimes")
            plt.axhline(
                y=time_counts.mean(),
                color="red",
                linestyle="--",
                label=f"Average: {time_counts.mean():.0f}",
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temporal_bias.png", dpi=300)
            plt.close()
            logger.info("Saved temporal bias visualization")

        # Plot 3: Severity distribution by district (heatmap)
        if "severity" in self.df.columns and "DISTRICT" in self.df.columns:
            plt.figure(figsize=(12, 8))

            # Create crosstab
            severity_by_district = (
                pd.crosstab(self.df["DISTRICT"], self.df["severity"], normalize="index") * 100
            )

            # Select top 10 districts by crime count
            top_districts = self.df["DISTRICT"].value_counts().head(10).index
            severity_by_district = severity_by_district.loc[top_districts]

            sns.heatmap(
                severity_by_district,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn_r",
                cbar_kws={"label": "Percentage (%)"},
            )
            plt.title("Severity Distribution by District (%)", fontsize=14, fontweight="bold")
            plt.xlabel("Crime Severity")
            plt.ylabel("District")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/severity_bias_heatmap.png", dpi=300)
            plt.close()
            logger.info("Saved severity bias heatmap")

        logger.info(f"All visualizations saved to {output_dir}")


def main():
    """Run complete bias detection analysis"""

    # Initialize detector
    detector = BiasDetector()

    # Generate comprehensive report
    bias_report = detector.generate_bias_report()

    # Create visualizations
    detector.visualize_bias()

    # Print summary
    print("\n" + "=" * 80)
    print("BIAS DETECTION SUMMARY")
    print("=" * 80)

    print("\n📊 Geographic Bias:")
    geo = bias_report.get("geographic_bias", {})
    print(f"  Overrepresented districts: {geo.get('overrepresented_districts', [])}")
    print(f"  Underrepresented districts: {geo.get('underrepresented_districts', [])}")

    print("\n⏰ Temporal Bias:")
    temp = bias_report.get("temporal_bias", {})
    print(f"  Biased time periods: {temp.get('biased_periods', [])}")

    print("\n⚖️ Mitigation Recommendations:")
    for i, rec in enumerate(bias_report.get("mitigation_recommendations", []), 1):
        print(f"  {i}. {rec}")

    print("\n✅ Bias detection complete!")
    print("📄 Full report: data/processed/bias_report.txt")
    print("📈 Visualizations: data/processed/visualizations/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
