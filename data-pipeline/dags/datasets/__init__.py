"""
Boston Pulse - Dataset DAGs

Airflow DAGs for each dataset's ETL pipeline.
Each DAG follows the standard pattern:
1. Ingest: Fetch data from external API
2. Validate Raw: Schema validation on raw data
3. Preprocess: Clean and transform data
4. Validate Processed: Schema validation on processed data
5. Build Features: Create analytics features
6. Validate Features: Final schema validation
7. Fairness Check: Bias detection
8. Commit: DVC versioning

Available DAGs:
    - crime_dag: Crime incident reports
    - service_311_dag: 311 service requests
    - vision_zero_dag: Traffic crash data
    - streetlight_dag: Streetlight locations
    - snow_emergency_dag: Snow emergency data
    - shots_fired_dag: Shots fired incidents
    - fire_dag: Fire incident reports
    - food_inspection_dag: Food establishment inspections
    - hospital_dag: Hospital locations
    - property_dag: Property assessments
    - emissions_dag: Building emissions data
    - mbta_dag: MBTA transit data
"""
