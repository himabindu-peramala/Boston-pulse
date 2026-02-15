"""
Boston Pulse - Aggregation DAGs

Airflow DAGs for cross-dataset aggregation:
- Combining multiple datasets for composite features
- Safety scores
- Transit safety metrics

Available DAGs:
    - safety_score_dag: Combine crime + 311 + streetlights
    - transit_safety_dag: Combine MBTA + crime near stops
    - neighborhood_summary_dag: Daily neighborhood-level summaries
"""
