# Crime Dataset

Reference implementation for the Boston Pulse crime incident data pipeline.

## Overview

This dataset contains crime incident reports from the Boston Police Department, sourced from the [Analyze Boston](https://data.boston.gov/dataset/crime-incident-reports) open data portal.

## Data Source

- **Source:** Boston Police Department Crime Incident Reports
- **API:** CKAN datastore_search (Analyze Boston)
- **Resource ID:** `12cb3883-56f5-47de-afa5-3b1cf61b257b`
- **Update Frequency:** Daily
- **Data Range:** 2015 - Present

## Pipeline Components

### 1. Ingester (`ingest.py`)

Fetches crime data from the Analyze Boston API.

```python
from src.datasets.crime import CrimeIngester

ingester = CrimeIngester()
result = ingester.run(execution_date="2024-01-15")
df = ingester.get_data()
```

**Features:**
- Watermark-based incremental ingestion
- Configurable batch size (default: 10,000 records)
- Automatic retry with exponential backoff
- Rate limiting compliance

### 2. Preprocessor (`preprocess.py`)

Cleans and validates raw crime data.

```python
from src.datasets.crime import CrimePreprocessor

preprocessor = CrimePreprocessor()
result = preprocessor.run(raw_df, execution_date="2024-01-15")
processed_df = preprocessor.get_data()
```

**Transformations:**
- Column renaming to standardized names
- Date/time parsing and validation
- Geographic coordinate validation (Boston bounds)
- Category standardization
- Shooting field conversion to boolean
- Duplicate removal

### 3. Feature Builder (`features.py`)

Creates aggregated crime features at the grid cell level.

```python
from src.datasets.crime import CrimeFeatureBuilder

builder = CrimeFeatureBuilder()
result = builder.run(processed_df, execution_date="2024-01-15")
features_df = builder.get_data()
```

**Features Generated:**
- Crime counts by time window (7d, 30d, 90d)
- Shooting incident counts and ratios
- Violent crime ratio
- Property crime ratio
- Night crime ratio
- Weekend crime ratio
- Normalized crime risk score

## Schema

### Raw Data (`schemas/crime/raw_schema.json`)

| Field | Type | Description |
|-------|------|-------------|
| INCIDENT_NUMBER | string | Primary key |
| OFFENSE_CODE | integer | Numeric offense code |
| OFFENSE_CODE_GROUP | string | Offense category |
| OFFENSE_DESCRIPTION | string | Detailed description |
| DISTRICT | string | Police district code |
| SHOOTING | string | Y/N indicator |
| OCCURRED_ON_DATE | datetime | When incident occurred |
| Lat | number | Latitude |
| Long | number | Longitude |

### Processed Data (`schemas/crime/processed_schema.json`)

| Field | Type | Description |
|-------|------|-------------|
| incident_number | string | Primary key |
| offense_code | integer | Numeric offense code |
| offense_category | string | Standardized category |
| district | string | Police district (uppercase) |
| shooting | boolean | Shooting involved |
| occurred_on_date | datetime | Validated datetime |
| lat | number | Validated latitude (Boston bounds) |
| long | number | Validated longitude (Boston bounds) |

### Features (`schemas/crime/features_schema.json`)

| Field | Type | Description |
|-------|------|-------------|
| grid_cell | string | Grid cell identifier |
| grid_lat | number | Grid center latitude |
| grid_long | number | Grid center longitude |
| district | string | Primary district |
| crime_count_7d | integer | Crimes in past 7 days |
| crime_count_30d | integer | Crimes in past 30 days |
| shooting_count_30d | integer | Shootings in past 30 days |
| violent_crime_ratio | number | Ratio of violent crimes |
| crime_risk_score | number | Normalized risk (0-1) |

## DAG

The crime pipeline DAG (`dags/datasets/crime_dag.py`) runs daily at 2 AM UTC.

### Pipeline Stages

1. **Ingest** - Fetch data from API
2. **Validate Raw** - Check raw schema
3. **Preprocess** - Clean and transform
4. **Validate Processed** - Check processed schema
5. **Build Features** - Create aggregated features
6. **Validate Features** - Check feature schema
7. **Detect Drift** - Check for distribution changes
8. **Check Fairness** - Evaluate fairness metrics
9. **Generate Model Card** - Create documentation
10. **Update Watermark** - Store ingestion progress

### Manual Trigger

```bash
airflow dags trigger crime_pipeline
```

## Configuration

Dataset-specific configuration in `configs/datasets/crime.yaml`:

```yaml
crime:
  schedule: "0 2 * * *"
  api:
    resource_id: "12cb3883-56f5-47de-afa5-3b1cf61b257b"
    batch_size: 10000
  watermark:
    field: "OCCURRED_ON_DATE"
    lookback_days: 7
  validation:
    strict_mode: true
    geo_bounds:
      min_lat: 42.2
      max_lat: 42.4
      min_lon: -71.2
      max_lon: -70.9
  features:
    grid_size: 0.001
    windows: [7, 30, 90]
  fairness:
    protected_attributes:
      - district
      - hour_of_day
```

## Testing

Run crime-specific tests:

```bash
# From data-pipeline directory
pytest tests/unit/datasets/crime/ -v

# With coverage
pytest tests/unit/datasets/crime/ -v --cov=src/datasets/crime
```

## Usage as Reference

This implementation serves as the reference for all other datasets. When adding a new dataset:

1. Copy the crime dataset structure
2. Implement the abstract methods from base classes
3. Create corresponding schemas
4. Create the DAG following the same pattern
5. Add comprehensive tests

## Data Quality Notes

- **Coordinates:** ~5% of records have missing or invalid coordinates
- **Dates:** Some historical records may have parsing issues
- **Categories:** Offense categories can change over time
- **Duplicates:** API may return duplicates on retry

## Contact

For issues with the crime dataset pipeline, contact the Tech Lead or create an issue in the repository.
