# Airflow DAG

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.decorators import dag
from airflow.models import Variable
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Variable.get(
    "oss_health_project_root",
    default_var="/Users/carolyn/Desktop/opensource/data/oss-health-data",
)

BACKEND_WEBHOOK_URL = Variable.get(
    "oss_health_backend_webhook_url",
    default_var="",
)


@dag(
    dag_id="oss_health_model_update_pipeline",
    description="Update reference set, detect drift, train challenger, promote model, and notify backend.",
    schedule="0 3 * * 1",
    start_date=pendulum.datetime(2026, 6, 8, tz="Asia/Seoul"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "oss-health-data",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["oss-health", "reference", "drift", "model-update"],
)
def oss_health_model_update_pipeline():
    BashOperator(
        task_id="run_model_update_pipeline",
        bash_command=f"""
        cd {PROJECT_ROOT} &&
        PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 src/model_update/run_update_pipeline.py \
          --project-root {PROJECT_ROOT} \
          --active-reference src/reference_store/active/reference_latest.csv \
          --scoring-reference src/reference_store/active/reference_latest.csv \
          --candidate-pool src/reference_store/candidates/candidate_pool.csv \
          --output-reference src/reference_store/active/reference_latest.csv \
          --output-candidate-pool src/reference_store/candidates/candidate_pool.csv \
          --drift-reference-features src/outputs/2_model/final_training_dataset.csv \
          --current-batch-features src/outputs/model_update/current_batch_features.csv \
          --base-model-path src/models/oss_health_best_model.joblib \
          --model-features src/models/oss_health_best_features.json \
          --model-metadata src/models/oss_health_model_metadata.json \
          --challenger-root src/model_registry/challenger \
          --champion-path src/model_registry/champion \
          --archive-root src/model_registry/archive \
          --backend-handoff-models-path src/backend_handoff/models \
          --data-models-path src/models \
          --backend-webhook-url "{BACKEND_WEBHOOK_URL}" \
          --replenish-candidates-on-demand
        """,
    )


oss_health_model_update_pipeline()