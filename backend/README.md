# Boston Pulse - MLOps Data Pipeline

## Project Overview

This repository contains a production-ready MLOps data pipeline for the Boston Pulse civic intelligence platform. The pipeline automates the ingestion, preprocessing, validation, and versioning of Boston crime incident data from the City of Boston Open Data Portal.

The pipeline demonstrates modern MLOps practices including automated workflow orchestration, data versioning, comprehensive testing, bias detection, and continuous monitoring.

## Team Members

- Himabindu Peramala
- Mukul Sridar Haladhi
- Prashanth Jaganathan
- Purvaja Narayana
- Sushma Ramesh
- Swetha Ganesh

## Dataset Information

### Crime Incident Reports (2023 - Present)

- **Source**: City of Boston Open Data Portal
- **Records**: 242,921 crime incidents
- **Date Range**: January 1, 2023 to January 26, 2026
- **Update Frequency**: Weekly updates from Boston Police Department
- **Format**: CSV with 17 base columns
- **Size**: Approximately 240 MB

### Key Features

The pipeline processes raw crime data and engineers the following features:
- Temporal features: year, month, day of week, hour, weekend indicator
- Time of day categories: Night, Morning, Afternoon, Evening
- Crime severity classification: Serious, Moderate, Minor
- District crime density metrics
- Coordinate validation flags
- Standardized text fields

## Repository Structure
```
data-pipeline/
├── dags/
│   └── crime_data_pipeline_dag.py    # Airflow DAG for automated pipeline
├── scripts/
│   ├── ingest_crime.py                # Data ingestion from Boston API
│   ├── preprocess_crime.py            # Data cleaning and feature engineering
│   └── bias_detection.py              # Bias detection and fairness analysis
├── tests/
│   ├── test_ingest_crime.py           # Unit tests for ingestion
│   └── test_preprocess_crime.py       # Unit tests for preprocessing
├── data/
│   ├── raw/                           # Raw downloaded data (tracked by DVC)
│   ├── processed/                     # Cleaned and processed data (tracked by DVC)
│   └── features/                      # Feature engineered datasets
├── logs/                              # Pipeline execution logs
├── requirements.txt                   # Python dependencies
├── dvc.yaml                           # DVC pipeline configuration
└── README.md                          # This file
```

## Installation and Setup

### Prerequisites

- Python 3.12 or higher
- Git installed and configured
- Virtual environment support

### Step 1: Clone the Repository
```bash
git clone https://github.com/himabindu-peramala/Boston-pulse.git
cd Boston-pulse/data-pipeline
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- requests: HTTP requests for API calls
- sodapy: Socrata Open Data API client
- pytest: Testing framework
- python-dotenv: Environment variable management
- dvc: Data version control
- apache-airflow: Workflow orchestration
- matplotlib, seaborn: Data visualization

### Step 5: Initialize DVC
```bash
cd ..  # Go to repository root
dvc init
dvc remote add -d myremote .dvc/storage
```

### Step 6: Pull Data
```bash
dvc pull
```

This downloads the versioned crime datasets to your local machine.

## Running the Pipeline

### Option 1: Run Individual Scripts

Execute pipeline components independently:

**Data Ingestion:**
```bash
cd data-pipeline
python scripts/ingest_crime.py
```

This downloads crime incident data from the Boston Open Data Portal and saves it to `data/raw/crime_data.csv`.

**Data Preprocessing:**
```bash
python scripts/preprocess_crime.py
```

This cleans the raw data, handles missing values, engineers features, and saves the processed data to `data/processed/crime_data_clean.csv`.

**Bias Detection:**
```bash
python scripts/bias_detection.py
```

This performs comprehensive bias analysis and generates a report at `data/processed/bias_report.txt` along with visualizations in `data/processed/visualizations/`.

### Option 2: Run Complete Pipeline via Airflow

**Initialize Airflow:**
```bash
export AIRFLOW_HOME=$(pwd)  # macOS/Linux
$env:AIRFLOW_HOME="$PWD"    # Windows PowerShell

airflow db init
```

**Test the DAG:**
```bash
airflow dags test crime_data_pipeline 2026-02-18
```

**Run the scheduler (for production):**
```bash
airflow scheduler
```

The DAG is scheduled to run weekly and executes the following tasks in sequence:
1. Download crime data from Boston API
2. Validate data quality
3. Preprocess and clean data
4. Generate statistics

## Running Tests

Execute the test suite to verify pipeline functionality:
```bash
pytest tests/ -v
```

Expected output: 12 tests passing

Test coverage includes:
- Data validation edge cases
- Duplicate detection
- Empty dataset handling
- Invalid coordinate detection
- Temporal feature extraction
- Text standardization
- Severity categorization
- Statistics generation

## Data Versioning with DVC

This pipeline uses DVC to version control large datasets separately from code.

### Tracking New Data

When crime data is updated:
```bash
dvc add data/raw/crime_data.csv
git add data/raw/crime_data.csv.dvc
git commit -m "Update crime data - February 2026"
```

### Accessing Historical Versions

To revert to a previous data version:
```bash
git checkout <commit-hash> data/raw/crime_data.csv.dvc
dvc checkout
```

### Cloud Storage Integration

For production deployment with Google Cloud Platform:
```bash
dvc remote add -d gcp gs://boston-pulse-data/dvc-storage
dvc push
```

Team members can then pull data from the shared cloud storage:
```bash
dvc pull
```

## Bias Detection Analysis

The bias detection module performs comprehensive fairness analysis across multiple dimensions:

### Geographic Bias

Analyzes crime distribution across Boston police districts to detect over-representation or under-representation.

**Key Findings:**
- District D4 is overrepresented with 35,215 incidents (14.5% of total)
- Districts A15, UNKNOWN, EXTERNAL show under-representation
- Average crimes per district: 16,195

**Mitigation**: Normalize crime safety scores by population density to avoid unfairly flagging high-density residential or commercial districts.

### Temporal Bias

Examines crime reporting patterns across different times of day.

**Key Findings:**
- Afternoon period: 83,344 incidents (34.3%)
- Morning period: 68,824 incidents (28.3%)
- Evening period: 47,965 incidents (19.7%)
- Night period: 42,788 incidents (17.6%)
- No significant temporal bias detected (all periods within expected ranges)

### Severity Bias

Analyzes whether certain districts have disproportionate serious crime classifications.

**Key Findings:**
- Overall distribution: Minor (49.2%), Moderate (27.8%), Serious (23.0%)
- One district identified with unusual severity pattern requiring further investigation
- Severity classifications appear generally consistent across districts

### Data Quality Bias

Identifies systematic gaps in data collection.

**Key Findings:**
- Three districts (A1, EXTERNAL, UNKNOWN) have more than 10% invalid coordinates
- 14,921 total records (6.1%) have invalid or missing geographic coordinates
- Coordinate quality varies by district, suggesting inconsistent data collection practices

**Mitigation**: Improve coordinate validation and collection processes in identified districts.

## Pipeline Monitoring

### Logging

All pipeline operations are logged to the `logs/` directory:
- `crime_ingestion.log`: Data download and validation logs
- `preprocessing.log`: Data cleaning and transformation logs
- `bias_detection.log`: Bias analysis execution logs

### Data Quality Metrics

The pipeline automatically tracks:
- Total records processed
- Missing value counts by column
- Duplicate record detection
- Coordinate validity rates
- Schema compliance

### Anomaly Detection

The validation module flags critical issues:
- Empty datasets
- High duplicate rates (>10%)
- Missing critical columns
- Invalid geographic coordinates (outside Boston bounds)
- High missing value percentages (>20%)

## Reproducibility

### Complete Reproduction Steps

Any user can reproduce the entire pipeline:

1. Clone the repository
2. Set up virtual environment and install dependencies
3. Run `dvc pull` to retrieve versioned datasets
4. Execute `python scripts/ingest_crime.py` to verify data acquisition
5. Execute `python scripts/preprocess_crime.py` to verify preprocessing
6. Run `pytest tests/` to verify all tests pass
7. Execute `airflow dags test crime_data_pipeline 2026-02-18` to verify DAG execution


