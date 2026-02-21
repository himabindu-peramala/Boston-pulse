"""
Boston Pulse - Maintenance DAGs

Airflow DAGs for maintenance tasks:
- DVC versioning (daily batch, weekly snapshot)
- Schema baseline updates
- Bias/fairness reports
- Data cleanup

Available DAGs:
    - dvc_daily_commit_dag: Daily DVC batch commit
    - dvc_weekly_snapshot_dag: Weekly full snapshot with tags
    - schema_update_dag: On-demand schema baseline updates
    - bias_report_dag: Weekly cross-dataset fairness summary
    - cleanup_dag: Temporary file cleanup
"""
