"""Airflow DAG: ingest the synthetic records into the service through POST /ingest.

The DAG only talks HTTP to the API (no direct ledger/vector-store access). Every request carries the
run context headers (X-Run-Id, X-Dag-Id, X-Task-Id, X-Try-Number, X-Data-Interval-Start) so a retry is
visible in the lineage. Config via env: CSPRO_API_URL (default http://127.0.0.1:8000) and
CSPRO_INGEST_KEY_<TENANT> (e.g. CSPRO_INGEST_KEY_ALPHA), normally injected from an Airflow Connection.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

try:  # pure logic lives next to the DAG; the dags folder is on sys.path inside Airflow
    from dags.ingest_logic import is_acceptable, load_records, post_manifest
except ImportError:
    from ingest_logic import is_acceptable, load_records, post_manifest

try:
    from airflow.decorators import dag, task
    from airflow.operators.python import get_current_context
except ImportError:  # airflow not installed: keep the module importable (static analysis / unit tests)
    def dag(**_kw):
        return lambda f: (lambda *a, **k: None)  # body never runs without Airflow

    def task(*_a, **_kw):
        return lambda f: f

    def get_current_context():
        return {}

DEFAULT_ARGS = {"owner": "data-eng", "retries": 2, "retry_delay": timedelta(seconds=30)}


@dag(
    dag_id="cspro_ingest",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["cspro", "ingest"],
)
def cspro_ingest():
    @task(retries=2)
    def ingest_all() -> list[dict]:
        import httpx

        context = get_current_context()
        ti = context["ti"]
        interval_start = context.get("data_interval_start")
        ctx = {
            "run_id": context["run_id"],
            "dag_id": ti.dag_id,
            "task_id": ti.task_id,
            "try_number": ti.try_number,
            "data_interval_start": interval_start.isoformat() if interval_start else None,
        }
        base_url = os.environ.get("CSPRO_API_URL", "http://127.0.0.1:8000")
        results = []
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            for record in load_records():
                key = os.environ[f"CSPRO_INGEST_KEY_{record['tenant'].upper()}"]
                resp = post_manifest(client, record, ctx, api_key=key)  # POST /ingest
                if not is_acceptable(resp):
                    raise RuntimeError(f"ingest failed for {record['doc_id']} v{record['version']}: {resp.status_code}")
                results.append({"doc_id": record["doc_id"], "version": record["version"], "status": resp.status_code})
        return results

    ingest_all()


dag_instance = cspro_ingest()
