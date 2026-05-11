"""
DAG de ingestão diária do DOU.

Roda seg-sex às 9h30 horário de Brasília (12:30 UTC).
Usa o venv do dou_chat montado em /opt/airflow/dou_chat.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "miguel",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="dou_ingest_daily",
    default_args=default_args,
    description="Ingestão diária do DOU (DO1, DO2, DO3)",
    schedule="30 12 * * 1-5",
    start_date=datetime(2026, 5, 1),
    catchup=False,
    tags=["dou_chat"],
) as dag:

    BashOperator(
        task_id="ingest_dou",
        bash_command=(
            "cd /opt/airflow/dou_chat && "
            "/opt/airflow/dou_chat/.venv/bin/python -m ingestion.pipeline --date {{ ds }}"
        ),
    )
