import os
import re

DAG_DIR = "/Users/himabinduperamala/Documents/Boston-pulse/data-pipeline/dags/datasets"

REPLACEMENTS = {
    r"from dags\.utils import get_effective_watermark, write_data": "from dags.utils.watermark import get_effective_watermark\n    from dags.utils.gcs_io import write_data",
    r"from dags\.utils import alert_anomaly_detected, read_data": "from dags.utils.alerting import alert_anomaly_detected\n    from dags.utils.gcs_io import read_data",
    r"from dags\.utils import alert_validation_failure, read_data": "from dags.utils.alerting import alert_validation_failure\n    from dags.utils.gcs_io import read_data",
    r"from dags\.utils import alert_preprocessing_complete, read_data, write_data": "from dags.utils.alerting import alert_preprocessing_complete\n    from dags.utils.gcs_io import read_data, write_data",
    r"from dags\.utils import read_data, write_data": "from dags.utils.gcs_io import read_data, write_data",
    r"from dags\.utils import alert_drift_detected, read_data": "from dags.utils.alerting import alert_drift_detected\n    from dags.utils.gcs_io import read_data",
    r"from dags\.utils import alert_fairness_violation, read_data": "from dags.utils.alerting import alert_fairness_violation\n    from dags.utils.gcs_io import read_data",
    r"from dags\.utils import alert_pipeline_complete": "from dags.utils.alerting import alert_pipeline_complete",
    r"from dags\.utils import record_pipeline_lineage": "from dags.utils.lineage_utils import record_pipeline_lineage",
    r"from dags\.utils import read_data, set_watermark": "from dags.utils.gcs_io import read_data\n    from dags.utils.watermark import set_watermark",
    r"from dags\.utils import read_data": "from dags.utils.gcs_io import read_data",
    r"from dags\.utils import on_dag_failure, on_dag_success, on_task_failure": "from dags.utils.callbacks import on_dag_failure, on_dag_success, on_task_failure",
}

for filename in os.listdir(DAG_DIR):
    if filename.endswith(".py"):
        filepath = os.path.join(DAG_DIR, filename)
        with open(filepath) as f:
            content = f.read()

        new_content = content
        for pattern, replacement in REPLACEMENTS.items():
            new_content = re.sub(pattern, replacement, new_content)

        if new_content != content:
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"Fixed imports in {filename}")
