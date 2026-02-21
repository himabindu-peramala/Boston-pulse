"""
Bias Detection and Fairness Analysis - Police Stations
Analyzes spatial equity and coverage bias in Boston Police Station distribution
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stations_bias_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StationsBiasDetector:
    """Detect spatial equity and coverage bias in police station distribution"""

    def __init__(self, data_path: str = "data/processed/police_stations_clean.csv"):
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} police station records")

        self.bias_report = {
            'geographic_coverage_bias': {},
            'size_equity_bias': {},
            'zone_distribution_bias': {},
            'mitigation_recommendations': []
        }

    def detect_geographic_coverage_bias(self):
        """
        Analyze whether stations are equitably distributed across neighborhoods
        """
        logger.info("Analyzing geographic coverage bias...")

        if 'NEIGHBORHOOD' not in self.df.columns:
            logger.warning("NEIGHBORHOOD column not found, skipping geographic analysis")
            return

        neighborhood_counts = self.df['NEIGHBORHOOD'].value_counts()
        total_stations = len(self.df)

        neighborhood_stats = []
        for neighborhood, count in neighborhood_counts.items():
            neighborhood_stats.append({
                'neighborhood': neighborhood,
                'station_count': count,
                'percentage': (count / total_stations) * 100
            })

        neighborhood_df = pd.DataFrame(neighborhood_stats).sort_values(
            'station_count', ascending=False
        )

        avg_stations = total_stations / len(neighborhood_counts)
        overrepresented = neighborhood_df[neighborhood_df['station_count'] > 1.5 * avg_stations]
        underrepresented = neighborhood_df[neighborhood_df['station_count'] < 0.5 * avg_stations]

        self.bias_report['geographic_coverage_bias'] = {
            'neighborhood_distribution': neighborhood_df.to_dict('records'),
            'avg_stations_per_neighborhood': avg_stations,
            'overrepresented': overrepresented['neighborhood'].tolist(),
            'underrepresented': underrepresented['neighborhood'].tolist(),
            'neighborhoods_with_no_station': []  # can be extended with census data
        }

        logger.info(f"Geographic bias complete:")
        logger.info(f"  Overrepresented: {overrepresented['neighborhood'].tolist()}")
        logger.info(f"  Underrepresented: {underrepresented['neighborhood'].tolist()}")

        return neighborhood_df

    def detect_size_equity_bias(self):
        """
        Analyze whether station sizes (sq footage) are equitably distributed
        across neighborhoods/zones — smaller stations may serve larger areas unfairly
        """
        logger.info("Analyzing station size equity bias...")

        if 'FT_SQFT' not in self.df.columns:
            logger.warning("FT_SQFT column not found, skipping size analysis")
            return

        size_stats = {
            'overall': {
                'mean_sqft': float(self.df['FT_SQFT'].mean()),
                'median_sqft': float(self.df['FT_SQFT'].median()),
                'min_sqft': float(self.df['FT_SQFT'].min()),
                'max_sqft': float(self.df['FT_SQFT'].max()),
            }
        }

        # Size distribution by zone
        if 'ZONE' in self.df.columns:
            zone_size = self.df.groupby('ZONE')['FT_SQFT'].agg(['mean', 'median', 'count'])
            size_stats['by_zone'] = zone_size.to_dict()

            # Flag zones where avg size deviates >30% from overall mean
            overall_mean = size_stats['overall']['mean_sqft']
            biased_zones = []
            for zone, row in zone_size.iterrows():
                deviation_pct = abs(row['mean'] - overall_mean) / overall_mean * 100
                if deviation_pct > 30:
                    biased_zones.append({
                        'zone': zone,
                        'avg_sqft': row['mean'],
                        'deviation_pct': deviation_pct
                    })

            size_stats['biased_zones'] = biased_zones
            if biased_zones:
                logger.warning(f"Found {len(biased_zones)} zones with disproportionate station sizes")
                self.bias_report['mitigation_recommendations'].append(
                    "Size Equity: Consider expanding under-resourced stations in zones with below-average sq footage"
                )

        self.bias_report['size_equity_bias'] = size_stats
        return size_stats

    def detect_zone_distribution_bias(self):
        """
        Analyze whether stations are clustered in Inner Boston vs outer neighborhoods
        """
        logger.info("Analyzing zone distribution bias...")

        if 'ZONE' not in self.df.columns:
            logger.warning("ZONE column not found, skipping zone analysis")
            return

        zone_counts = self.df['ZONE'].value_counts()
        total = len(self.df)

        zone_stats = []
        for zone, count in zone_counts.items():
            zone_stats.append({
                'zone': zone,
                'station_count': count,
                'percentage': (count / total) * 100
            })

        zone_df = pd.DataFrame(zone_stats)

        # Expected roughly equal distribution across zones
        expected_pct = 100 / len(zone_counts)
        zone_df['deviation'] = abs(zone_df['percentage'] - expected_pct)
        biased_zones = zone_df[zone_df['deviation'] > 15]

        self.bias_report['zone_distribution_bias'] = {
            'zone_distribution': zone_df.to_dict('records'),
            'expected_pct_per_zone': expected_pct,
            'biased_zones': biased_zones['zone'].tolist()
        }

        logger.info(f"Zone distribution bias: {biased_zones['zone'].tolist()}")

        if not biased_zones.empty:
            self.bias_report['mitigation_recommendations'].append(
                "Zone Bias: Outer zones appear underserved — evaluate coverage gaps for equitable deployment"
            )

        return zone_df

    def detect_coordinate_quality_bias(self):
        """
        Analyze whether certain neighborhoods have poor coordinate data
        """
        logger.info("Analyzing coordinate quality bias...")

        if 'VALID_COORDS' not in self.df.columns:
            logger.warning("VALID_COORDS column not found, skipping coordinate analysis")
            return

        if 'NEIGHBORHOOD' in self.df.columns:
            coord_quality = self.df.groupby('NEIGHBORHOOD')['VALID_COORDS'].agg(
                total='count',
                valid='sum'
            )
            coord_quality['invalid'] = coord_quality['total'] - coord_quality['valid']
            coord_quality['invalid_pct'] = (coord_quality['invalid'] / coord_quality['total']) * 100

            problematic = coord_quality[coord_quality['invalid_pct'] > 0]
            if not problematic.empty:
                logger.warning(f"Neighborhoods with invalid coordinates: {problematic.index.tolist()}")
                self.bias_report['mitigation_recommendations'].append(
                    "Data Quality: Fix missing/invalid coordinates for accurate spatial coverage analysis"
                )

    def generate_bias_report(self, output_path: str = "data/processed/stations_bias_report.txt"):
        """Generate comprehensive bias detection report for police stations"""
        logger.info("Generating stations bias detection report...")

        self.detect_geographic_coverage_bias()
        self.detect_size_equity_bias()
        self.detect_zone_distribution_bias()
        self.detect_coordinate_quality_bias()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BIAS DETECTION REPORT - BOSTON POLICE STATIONS\n")
            f.write("=" * 80 + "\n\n")

            # Geographic Coverage Bias
            f.write("1. GEOGRAPHIC COVERAGE BIAS\n")
            f.write("-" * 80 + "\n")
            geo = self.bias_report.get('geographic_coverage_bias', {})
            if geo:
                f.write(f"Avg stations per neighborhood: {geo.get('avg_stations_per_neighborhood', 0):.2f}\n\n")
                f.write("Overrepresented Neighborhoods:\n")
                for n in geo.get('overrepresented', []):
                    f.write(f"  - {n}\n")
                f.write("\nUnderrepresented Neighborhoods:\n")
                for n in geo.get('underrepresented', []):
                    f.write(f"  - {n}\n")
                f.write("\nStation Distribution by Neighborhood:\n")
                for item in geo.get('neighborhood_distribution', []):
                    f.write(f"  {item['neighborhood']}: {item['station_count']} ({item['percentage']:.1f}%)\n")
            f.write("\n")

            # Size Equity Bias
            f.write("2. STATION SIZE EQUITY BIAS\n")
            f.write("-" * 80 + "\n")
            size = self.bias_report.get('size_equity_bias', {})
            if size:
                overall = size.get('overall', {})
                f.write(f"Mean sq footage: {overall.get('mean_sqft', 0):.0f} ft²\n")
                f.write(f"Median sq footage: {overall.get('median_sqft', 0):.0f} ft²\n")
                f.write(f"Range: {overall.get('min_sqft', 0):.0f} – {overall.get('max_sqft', 0):.0f} ft²\n")
                if size.get('biased_zones'):
                    f.write("\nZones with Disproportionate Station Sizes:\n")
                    for z in size['biased_zones']:
                        f.write(f"  {z['zone']}: {z['avg_sqft']:.0f} ft² (deviation: {z['deviation_pct']:.1f}%)\n")
            f.write("\n")

            # Zone Distribution Bias
            f.write("3. ZONE DISTRIBUTION BIAS (Inner / Mid / Outer Boston)\n")
            f.write("-" * 80 + "\n")
            zone = self.bias_report.get('zone_distribution_bias', {})
            if zone:
                f.write(f"Expected distribution per zone: {zone.get('expected_pct_per_zone', 33.3):.1f}%\n\n")
                for item in zone.get('zone_distribution', []):
                    deviation = item.get('deviation', 0)
                    status = "⚠️ BIASED" if deviation > 15 else "✓ Normal"
                    f.write(f"  {item['zone']}: {item['station_count']} stations ({item['percentage']:.1f}%) [{status}]\n")
            f.write("\n")

            # Recommendations
            f.write("4. BIAS MITIGATION RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            recommendations = self.bias_report.get('mitigation_recommendations', [])
            if not recommendations:
                recommendations.append("No significant bias detected requiring mitigation")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF BIAS DETECTION REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Bias report saved to {output_path}")
        return self.bias_report

    def visualize_bias(self, output_dir: str = "data/processed/stations_visualizations"):
        """Create visualizations of detected bias"""
        logger.info("Creating bias visualizations...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Plot 1: Station count by neighborhood
        if 'NEIGHBORHOOD' in self.df.columns:
            plt.figure(figsize=(12, 6))
            counts = self.df['NEIGHBORHOOD'].value_counts()
            counts.plot(kind='bar', color='steelblue')
            plt.title('Police Stations by Neighborhood', fontsize=14, fontweight='bold')
            plt.xlabel('Neighborhood')
            plt.ylabel('Number of Stations')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/neighborhood_distribution.png", dpi=300)
            plt.close()

        # Plot 2: Zone distribution pie chart
        if 'ZONE' in self.df.columns:
            plt.figure(figsize=(8, 8))
            zone_counts = self.df['ZONE'].value_counts()
            plt.pie(
                zone_counts.values,
                labels=zone_counts.index,
                autopct='%1.1f%%',
                colors=['#4ECDC4', '#FF6B6B', '#FFE66D'],
                startangle=140
            )
            plt.title('Station Distribution: Inner / Mid / Outer Boston', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/zone_distribution.png", dpi=300)
            plt.close()

        # Plot 3: Building size by zone boxplot
        if 'FT_SQFT' in self.df.columns and 'ZONE' in self.df.columns:
            plt.figure(figsize=(10, 6))
            self.df.boxplot(column='FT_SQFT', by='ZONE', grid=False)
            plt.title('Station Size (sq ft) by Zone', fontsize=14, fontweight='bold')
            plt.suptitle('')
            plt.xlabel('Zone')
            plt.ylabel('Square Footage')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/size_by_zone_boxplot.png", dpi=300)
            plt.close()

        logger.info(f"Visualizations saved to {output_dir}")


def main():
    detector = StationsBiasDetector()
    bias_report = detector.generate_bias_report()
    detector.visualize_bias()

    print("\n" + "=" * 80)
    print("STATIONS BIAS DETECTION SUMMARY")
    print("=" * 80)

    geo = bias_report.get('geographic_coverage_bias', {})
    print(f"\n📍 Geographic Coverage Bias:")
    print(f"  Overrepresented: {geo.get('overrepresented', [])}")
    print(f"  Underrepresented: {geo.get('underrepresented', [])}")

    zone = bias_report.get('zone_distribution_bias', {})
    print(f"\n🗺️  Zone Bias:")
    print(f"  Biased zones: {zone.get('biased_zones', [])}")

    print(f"\n✅ Mitigation Recommendations:")
    for i, rec in enumerate(bias_report.get('mitigation_recommendations', []), 1):
        print(f"  {i}. {rec}")

    print(f"\n📄 Full report: data/processed/stations_bias_report.txt")
    print(f"📈 Visualizations: data/processed/stations_visualizations/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()