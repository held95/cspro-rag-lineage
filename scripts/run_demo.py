"""Recording-friendly demo: prints a readable transcript of the behaviours the assessment looks for.

    python -m scripts.run_demo [--embed ollama|hash]

The app runs in-process (FastAPI TestClient) with a temporary ledger and in-memory Qdrant, so nothing is written to
the repository. With the default `--embed ollama` it talks to the local Ollama (models must be pulled). The LLM is always Ollama (the fake LLM exists only in tests);
`--embed hash` only replaces the embedder, which gives poor retrieval quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.container import build_container
from app.core.config import KeyRecord, Settings
from app.main import create_app

KEYS = {
    "demo-alpha-admin": KeyRecord(tenant_id="alpha", subject="alpha-admin", groups=["support", "oncall"],
                                  scopes=["docs:read", "docs:write", "ask:read"]),
    "demo-alpha-support": KeyRecord(tenant_id="alpha", subject="alpha-support", groups=["support"],
                                    scopes=["docs:read", "ask:read"]),
    "demo-beta-admin": KeyRecord(tenant_id="beta", subject="beta-admin", groups=[],
                                 scopes=["docs:read", "docs:write", "ask:read"]),
}


def title(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show(label: str, r) -> None:
    body = r.json()
    print(f"-> {label}: HTTP {r.status_code}")
    print(json.dumps(body, indent=2, ensure_ascii=False)[:1600])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", choices=["ollama", "hash"], default="ollama")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="cspro_demo_"))
    settings = Settings(ledger_path=str(tmp / "ledger.db"), vector_path=None, embed_backend=a.embed,
                        api_keys={hashlib.sha256(k.encode()).hexdigest(): v for k, v in KEYS.items()})
    container = build_container(settings)
    client = TestClient(create_app(settings, container), raise_server_exceptions=False)

    from tests.helpers import ingest_records  # the same 15 synthetic record versions used by the tests
    ingest_records(SimpleNamespace(service=container.ingestion), "full")

    def h(key, **extra):
        return {"X-API-Key": key, **extra}

    def ask(key, q):
        return client.post("/ask", json={"question": q}, headers=h(key))

    title(f"0. Health  (embed={a.embed}, llm=ollama, model={settings.llm_model})")
    show("GET /health", client.get("/health"))

    title("1. The lineage failure: a retry replays an OLD version after a newer one was accepted")
    doc = {"doc_id": "KB-900", "kind": "kb", "title": "Demo handbook"}
    v1 = {**doc, "version": 1, "content": "Demo handbook v1. The support office opens at 9 am."}
    v2 = {**doc, "version": 2, "content": "Demo handbook v2. The support office opens at 10 am."}
    show("POST /ingest v1", client.post("/ingest", json=v1, headers=h("demo-alpha-admin")))
    show("POST /ingest v2", client.post("/ingest", json=v2, headers=h("demo-alpha-admin")))
    show("POST /ingest v1 AGAIN (Airflow retry, try 2)",
         client.post("/ingest", json=v1, headers=h("demo-alpha-admin", **{
             "X-Run-Id": "demo-run-1", "X-Dag-Id": "cspro_ingest", "X-Task-Id": "post_manifest", "X-Try-Number": "2"})))
    r = client.get("/documents/KB-900", headers=h("demo-alpha-admin"))
    print(f"-> still served: version={r.json()['version']} ({r.json()['content']!r})")
    lineage = client.get("/documents/KB-900/lineage", headers=h("demo-alpha-admin")).json()
    print("-> lineage:", json.dumps(lineage, ensure_ascii=False)[:900])

    title("2. Cited answer, abstention, conflict")
    show("refund question", ask("demo-alpha-admin", "What is the refund window?"))
    show("unrelated question", ask("demo-alpha-admin", "Who won the 1998 football world cup?"))
    show("conflicting runbooks (KB-003 vs KB-004)",
         ask("demo-alpha-admin", "What is the first response time for a P1 incident? Is there any conflicting information?"))

    title("3. Prompt injection in a ticket (TCK-1002) and tenant isolation")
    r = ask("demo-alpha-admin", "What was ticket 1002 about? Is the parcel late?")
    show("injection question", r)
    print(f"-> beta canary in response: {'CANARY-BETA-7f3a9' in r.text}   (must be False)")
    r = client.get("/documents/KB-101", headers=h("demo-alpha-admin"))
    show("alpha asks for beta's KB-101", r)
    r2 = client.get("/documents/KB-NOPE", headers=h("demo-alpha-admin"))
    print(f"-> same body as a document that does not exist: {r.json()['error']['message'] == r2.json()['error']['message']}")
    show("alpha-support (no oncall group) asks for the restricted runbook",
         client.get("/documents/KB-003", headers=h("demo-alpha-support")))

    title("4. Next: run the tests live")
    print("python -m pytest -m failure_repro -v    # naive pipeline XFAIL, fixed pipeline PASS")
    print("python -m pytest -m security -q          # includes SEC-01 (obedient LLM) and SEC-24 (controls off -> leak)")
    print("python -m pytest -q                      # full suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
