"""Pure ingest logic used by the Airflow DAG. No airflow import, no app import: testable anywhere."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "records.json"

# Fields of a synthetic record that are NOT part of DocumentManifest (the tenant comes from the API key).
_NON_MANIFEST = {"tenant", "stages", "role"}
INGEST_PATH = "/ingest"


def load_records(path: Path | str = RECORDS_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text("utf-8"))


def manifest_body(record: dict[str, Any]) -> dict[str, Any]:
    """Record -> DocumentManifest JSON body (extra='forbid' on the server, so strip helper fields)."""
    return {k: v for k, v in record.items() if k not in _NON_MANIFEST and v is not None}


def run_headers(ctx: dict[str, Any]) -> dict[str, str]:
    """Orchestrator lineage headers read by app/api/deps.py get_run_context."""
    mapping = {
        "X-Run-Id": ctx.get("run_id"),
        "X-Dag-Id": ctx.get("dag_id"),
        "X-Task-Id": ctx.get("task_id"),
        "X-Try-Number": ctx.get("try_number"),
        "X-Data-Interval-Start": ctx.get("data_interval_start"),
    }
    return {k: str(v) for k, v in mapping.items() if v is not None}


def post_manifest(client: Any, record: dict[str, Any], ctx: dict[str, Any], api_key: str | None = None):
    """POST one record to /ingest with run-context headers. ``ctx['api_key']`` is used if api_key is None.

    ``client`` is anything with ``.post(url, json=..., headers=...)`` (httpx.Client or a TestClient).
    Returns the response; interpretation is left to ``is_acceptable``.
    """
    key = api_key or ctx.get("api_key")
    headers = {**run_headers(ctx), "X-API-Key": key} if key else run_headers(ctx)
    return client.post(INGEST_PATH, json=manifest_body(record), headers=headers)


def is_acceptable(response: Any) -> bool:
    """2xx is success. 409 STALE_VERSION on a retry is also fine: a newer version already won (do not retry)."""
    if 200 <= response.status_code < 300:
        return True
    if response.status_code == 409:
        try:
            return response.json().get("error", {}).get("code") == "STALE_VERSION"
        except ValueError:
            return False
    return False
