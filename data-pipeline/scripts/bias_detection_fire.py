"""
Bias Detection and Fairness Analysis - Fire Incidents
Analyzes temporal, geographic, and severity bias in Boston Fire Incident data
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/fire_bias_detection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FireBiasDetector:
    """Detect bias in fire incident data across geographic and temporal dimensions"""

    def __init__(self, data_path: str = "data/processed/fire_incidents_clean.csv"):
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} fire incident records")

        self.bias_report = {
            "geographic_bias": {},
            "temporal_bias": {},
            "severity_bias": {},
            "loss_bias": {},
            "mitigation_recommendations": [],
        }

    def detect_geographic_bias(self):
        """Analyze bias across districts and neighborhoods"""
        logger.info("Analyzing geographic bias...")

        if "District" not in self.df.columns:
            logger.warning("District column not found, skipping geographic bias")
            return

        district_counts = self.df["District"].value_counts()
        total = len(self.df)

        district_stats = []
        for district, count in district_counts.items():
            district_stats.append(
                {"district": district, "incident_count": count, "percentage": (count / total) * 100}
            )

        district_df = pd.DataFrame(district_stats).sort_values("incident_count", ascending=False)

        avg = total / len(district_counts)
        overrepresented = district_df[district_df["incident_count"] > 2 * avg]
        underrepresented = district_df[district_df["incident_count"] < 0.5 * avg]

        self.bias_report["geographic_bias"] = {
            "district_distribution": district_df.to_dict("records"),
            "avg_incidents_per_district": avg,
            "overrepresented_districts": overrepresented["district"].tolist(),
            "underrepresented_districts": underrepresented["district"].tolist(),
        }

        logger.info(f"Geographic bias: overrepresented: {
                overrepresented['district'].tolist()}")
        return district_df

    def detect_temporal_bias(self):
        """Analyze bias across time periods"""
        logger.info("Analyzing temporal bias...")

        if "time_of_day" not in self.df.columns:
            logger.warning("time_of_day column not found, skipping temporal bias")
            return

        time_distribution = self.df["time_of_day"].value_counts()
        total = len(self.df)
        expected_pct = 25.0

        time_stats = []
        for period, count in time_distribution.items():
            pct = (count / total) * 100
            time_stats.append(
                {
                    "time_period": period,
                    "incident_count": count,
                    "percentage": pct,
                    "deviation": abs(pct - expected_pct),
                }
            )

        time_df = pd.DataFrame(time_stats).sort_values("incident_count", ascending=False)
        biased_periods = time_df[time_df["deviation"] > 10]

        self.bias_report["temporal_bias"] = {
            "time_distribution": time_df.to_dict("records"),
            "biased_periods": biased_periods["time_period"].tolist(),
            "expected_percentage": expected_pct,
        }

        logger.info(f"Temporal bias: biased periods: {
                biased_periods['time_period'].tolist()}")
        return time_df

    def detect_severity_bias(self):
        """Analyze whether certain districts have disproportionate severity"""
        logger.info("Analyzing severity bias...")

        if "severity_category" not in self.df.columns:
            logger.warning("severity_category column not found, skipping severity bias")
            return

        severity_distribution = self.df["severity_category"].value_counts()
        total = len(self.df)

        severity_stats = {"overall_distribution": {}}
        for severity, count in severity_distribution.items():
            severity_stats["overall_distribution"][severity] = {
                "count": int(count),
                "percentage": (count / total) * 100,
            }

        if "District" in self.df.columns:
            district_severity = (
                pd.crosstab(self.df["District"], self.df["severity_category"], normalize="index")
                * 100
            )

            severity_stats["by_district"] = district_severity.to_dict()

            if "Fire" in district_severity.columns:
                overall_fire_pct = (
                    severity_stats["overall_distribution"].get("Fire", {}).get("percentage", 0)
                )
                biased_districts = []
                for district in district_severity.index:
                    district_fire_pct = district_severity.loc[district, "Fire"]
                    deviation = abs(district_fire_pct - overall_fire_pct)
                    if deviation > 15:
                        biased_districts.append(
                            {
                                "district": district,
                                "fire_pct": district_fire_pct,
                                "deviation": deviation,
                            }
                        )
                severity_stats["biased_districts"] = biased_districts

                if biased_districts:
                    self.bias_report["mitigation_recommendations"].append(
                        "Severity Bias: Some districts have disproportionate actual fire rates — review resource allocation"
                    )

        self.bias_report["severity_bias"] = severity_stats
        return severity_stats

    def detect_loss_bias(self):
        """Analyze whether financial losses are distributed equitably across districts"""
        logger.info("Analyzing financial loss bias...")

        if "total_loss" not in self.df.columns or "District" not in self.df.columns:
            logger.warning("total_loss or District column not found, skipping loss bias")
            return

        loss_by_district = (
            self.df.groupby("District")["total_loss"]
            .agg(total="sum", mean="mean", median="median", count="count")
            .sort_values("total", ascending=False)
        )

        overall_mean = self.df["total_loss"].mean()
        biased_districts = []
        for district, row in loss_by_district.iterrows():
            if overall_mean > 0:
                deviation_pct = abs(row["mean"] - overall_mean) / overall_mean * 100
                if deviation_pct > 50:
                    biased_districts.append(
                        {
                            "district": district,
                            "mean_loss": row["mean"],
                            "deviation_pct": deviation_pct,
                        }
                    )

        self.bias_report["loss_bias"] = {
            "loss_by_district": loss_by_district.to_dict(),
            "overall_mean_loss": overall_mean,
            "biased_districts": biased_districts,
        }

        if biased_districts:
            self.bias_report["mitigation_recommendations"].append(
                "Loss Bias: Districts with disproportionate financial losses may need additional fire prevention resources"
            )

        logger.info(f"Loss bias analysis complete: {
                len(biased_districts)} districts flagged")
        return loss_by_district

    def generate_bias_report(self, output_path: str = "data/processed/fire_bias_report.txt"):
        """Generate comprehensive bias detection report for fire incidents"""
        logger.info("Generating fire incident bias detection report...")

        self.detect_geographic_bias()
        self.detect_temporal_bias()
        self.detect_severity_bias()
        self.detect_loss_bias()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("BIAS DETECTION REPORT - BOSTON FIRE INCIDENTS\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. GEOGRAPHIC BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")
            geo = self.bias_report.get("geographic_bias", {})
            if geo:
                f.write(f"Avg incidents per district: {
                        geo.get(
                            'avg_incidents_per_district',
                            0):.0f}\n\n")
                f.write("Overrepresented Districts (>2x average):\n")
                for d in geo.get("overrepresented_districts", []):
                    f.write(f"  - {d}\n")
                f.write("\nUnderrepresented Districts (<0.5x average):\n")
                for d in geo.get("underrepresented_districts", []):
                    f.write(f"  - {d}\n")
                f.write("\nTop 5 Districts by Incident Count:\n")
                for item in geo.get("district_distribution", [])[:5]:
                    f.write(f"  {
                            item['district']}: {
                            item['incident_count']} ({
                            item['percentage']:.2f}%)\n")
            f.write("\n")

            f.write("2. TEMPORAL BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")
            temp = self.bias_report.get("temporal_bias", {})
            if temp:
                f.write(f"Expected per time period: {
                        temp.get(
                            'expected_percentage',
                            25):.1f}%\n\n")
                for item in temp.get("time_distribution", []):
                    deviation = item.get("deviation", 0)
                    status = "⚠️ BIASED" if deviation > 10 else "✓ Normal"
                    f.write(f"  {
                            item['time_period']}: {
                            item['incident_count']} ({
                            item['percentage']:.2f}%) [{status}]\n")
            f.write("\n")

            f.write("3. SEVERITY BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")
            sev = self.bias_report.get("severity_bias", {})
            if sev:
                f.write("Overall Severity Distribution:\n")
                for severity, stats in sev.get("overall_distribution", {}).items():
                    f.write(f"  {severity}: {
                            stats['count']} ({
                            stats['percentage']:.2f}%)\n")
            f.write("\n")

            f.write("4. FINANCIAL LOSS BIAS ANALYSIS\n")
            f.write("-" * 80 + "\n")
            loss = self.bias_report.get("loss_bias", {})
            if loss:
                f.write(f"Overall mean loss per incident: ${
                        loss.get(
                            'overall_mean_loss',
                            0):,.0f}\n\n")
                if loss.get("biased_districts"):
                    f.write("Districts with Disproportionate Losses:\n")
                    for item in loss["biased_districts"]:
                        f.write(f"  {
                                item['district']}: ${
                                item['mean_loss']:,.0f} mean loss (deviation: {
                                item['deviation_pct']:.1f}%)\n")
            f.write("\n")

            f.write("5. BIAS MITIGATION RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            recommendations = self.bias_report.get("mitigation_recommendations", [])
            geo = self.bias_report.get("geographic_bias", {})
            if geo.get("overrepresented_districts"):
                recommendations.append(
                    "Geographic Bias: Normalize incident scores by population density"
                )
            temp = self.bias_report.get("temporal_bias", {})
            if temp.get("biased_periods"):
                recommendations.append(
                    "Temporal Bias: Account for reporting delays in nighttime incidents"
                )
            if not recommendations:
                recommendations.append("No significant bias detected requiring mitigation")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF BIAS DETECTION REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Bias report saved to {output_path}")
        return self.bias_report

    def visualize_bias(self, output_dir: str = "data/processed/fire_visualizations"):
        """Create visualizations of detected bias"""
        logger.info("Creating fire bias visualizations...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if "District" in self.df.columns:
            plt.figure(figsize=(12, 6))
            counts = self.df["District"].value_counts().head(10)
            counts.plot(kind="bar", color="firebrick")
            plt.title("Fire Incidents by District (Top 10)", fontsize=14, fontweight="bold")
            plt.xlabel("District")
            plt.ylabel("Number of Incidents")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/geographic_bias.png", dpi=300)
            plt.close()

        if "severity_category" in self.df.columns:
            plt.figure(figsize=(10, 6))
            counts = self.df["severity_category"].value_counts()
            counts.plot(kind="bar", color="darkorange")
            plt.title("Fire Incident Severity Distribution", fontsize=14, fontweight="bold")
            plt.xlabel("Severity Category")
            plt.ylabel("Number of Incidents")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/severity_distribution.png", dpi=300)
            plt.close()

        if "time_of_day" in self.df.columns:
            plt.figure(figsize=(8, 6))
            counts = self.df["time_of_day"].value_counts()
            colors = ["#FF6B6B" if c > counts.mean() * 1.2 else "#4ECDC4" for c in counts.values]
            counts.plot(kind="bar", color=colors)
            plt.title("Fire Incidents by Time of Day", fontsize=14, fontweight="bold")
            plt.xlabel("Time Period")
            plt.ylabel("Number of Incidents")
            plt.axhline(y=counts.mean(), color="red", linestyle="--", label=f"Average: {
                    counts.mean():.0f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temporal_bias.png", dpi=300)
            plt.close()

        logger.info(f"Visualizations saved to {output_dir}")


def main():
    detector = FireBiasDetector()
    bias_report = detector.generate_bias_report()
    detector.visualize_bias()

    print("\n" + "=" * 80)
    print("FIRE INCIDENTS BIAS DETECTION SUMMARY")
    print("=" * 80)

    geo = bias_report.get("geographic_bias", {})
    print("\n📍 Geographic Bias:")
    print(f"  Overrepresented: {geo.get('overrepresented_districts', [])}")
    print(f"  Underrepresented: {geo.get('underrepresented_districts', [])}")

    temp = bias_report.get("temporal_bias", {})
    print("\n⏰ Temporal Bias:")
    print(f"  Biased time periods: {temp.get('biased_periods', [])}")

    print("\n✅ Mitigation Recommendations:")
    for i, rec in enumerate(bias_report.get("mitigation_recommendations", []), 1):
        print(f"  {i}. {rec}")

    print("\n📄 Full report: data/processed/fire_bias_report.txt")
    print("📈 Visualizations: data/processed/fire_visualizations/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
