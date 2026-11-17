"""LIN-04 (static AST check of dags/ingest_dag.py) and a LIN-03-like retry replay via the pure DAG logic.

IMPORTANT: no Airflow scheduler/executor/DagBag is run here and Airflow is not installed. The DAG file is
only parsed with ``ast`` (never imported). The replay drives ``dags/ingest_logic.py`` (the code the DAG
task calls) against the in-process test app, simulating the headers of a second try (try_number=2).
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from tests.conftest import build_app

pytestmark = pytest.mark.unit

DAGS = Path(__file__).resolve().parents[2] / "dags"
TREE = ast.parse((DAGS / "ingest_dag.py").read_text("utf-8"))


def _load_logic():
    spec = importlib.util.spec_from_file_location("ingest_logic_under_test", DAGS / "ingest_logic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


logic = _load_logic()


def _constants(tree) -> list[str]:
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_lin04_retries_is_two():
    values = [kw.value.value for n in ast.walk(TREE)
              for kw in (n.keywords if isinstance(n, ast.Call) else []) if kw.arg == "retries"]
    values += [v.value for n in ast.walk(TREE) if isinstance(n, ast.Dict)
               for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == "retries"]
    assert values and all(v == 2 for v in values)


def test_lin04_uses_taskflow_decorators():
    names = set()
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                names.add(getattr(target, "id", getattr(target, "attr", None)))
    assert {"dag", "task"} <= names


def test_lin04_calls_ingest_endpoint_and_has_no_direct_store_access():
    assert logic.INGEST_PATH == "/ingest"
    called = {n.func.id for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "post_manifest" in called  # the task goes through the pure helper that POSTs /ingest
    imported = set()
    for n in ast.walk(TREE):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module or "")
    assert not any(m.startswith("app") or "ledger" in m or "qdrant" in m for m in imported)
    assert "/ingest" in (DAGS / "ingest_logic.py").read_text("utf-8")


def test_run_headers_use_names_the_app_reads():
    h = logic.run_headers({"run_id": "r1", "dag_id": "d", "task_id": "t", "try_number": 2,
                           "data_interval_start": "2025-01-01T00:00:00+00:00"})
    assert h == {"X-Run-Id": "r1", "X-Dag-Id": "d", "X-Task-Id": "t", "X-Try-Number": "2",
                 "X-Data-Interval-Start": "2025-01-01T00:00:00+00:00"}


def test_manifest_body_strips_non_manifest_fields():
    rec = logic.load_records()[0]
    body = logic.manifest_body(rec)
    assert not {"tenant", "stages", "role"} & set(body)


def test_lin03_like_retry_after_newer_version_is_stale(tmp_path):
    app = build_app(tmp_path)
    recs = {r["version"]: r for r in logic.load_records() if r["tenant"] == "alpha" and r["doc_id"] == "KB-001"}
    key = "alpha-admin-key"
    ctx = {"run_id": "manual__1", "dag_id": "cspro_ingest", "task_id": "ingest_all", "try_number": 1}

    assert logic.post_manifest(app.client, recs[1], ctx, api_key=key).status_code == 201
    assert logic.post_manifest(app.client, recs[2], ctx, api_key=key).status_code == 201

    retry = logic.post_manifest(app.client, recs[1], {**ctx, "try_number": 2}, api_key=key)
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "STALE_VERSION"
    assert logic.is_acceptable(retry)  # the DAG treats it as done, not as a failure to retry forever

    assert app.stack.ledger.get_active_version("alpha", "KB-001") == 2
    lineage = app.client.get("/documents/KB-001/lineage", headers=app.h(key)).json()
    assert lineage["active_version"] == 2
    stale = [e for e in lineage["events"] if e["action"] == "rejected_stale"]
    assert stale and stale[-1]["try_number"] == 2 and stale[-1]["run_id"] == "manual__1"


def test_is_acceptable_rejects_other_errors(tmp_path):
    app = build_app(tmp_path)
    r = logic.post_manifest(app.client, logic.load_records()[0], {"run_id": "x"}, api_key="wrong")
    assert r.status_code == 401 and not logic.is_acceptable(r)
