"""HTTP demo against a RUNNING server (what the recording should show).

    python -m scripts.seed_demo --embed ollama                      # once: writes .env.local, prints the keys
    python -m uvicorn app.main:create_app --factory                 # terminal 1
    python -m scripts.api_demo --alpha-key <k> --support-key <k> --beta-key <k>      # terminal 2

Keys can also come from CSPRO_DEMO_ALPHA_KEY / CSPRO_DEMO_SUPPORT_KEY / CSPRO_DEMO_BETA_KEY.
Only HTTP requests are made; nothing is imported from the app.
"""
from __future__ import annotations

import argparse
import json
import os

import httpx


def show(title: str, r: httpx.Response, keys=("status", "answer", "reason", "error")) -> dict:
    try:
        body = r.json()
    except ValueError:
        body = {}
    print(f"\n### {title}\n-> HTTP {r.status_code}")
    for k in keys:
        if k in body:
            print(f"   {k}: {json.dumps(body[k], ensure_ascii=False)[:300]}")
    for c in body.get("citations", []):
        print(f"   cite {c['source_id']} -> {c['doc_id']} v{c['version']}: {c['excerpt'][:110]!r}")
    if body.get("conflict", {}).get("detected"):
        print(f"   conflict between: {body['conflict']['doc_ids']}")
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--alpha-key", default=os.environ.get("CSPRO_DEMO_ALPHA_KEY"))
    ap.add_argument("--support-key", default=os.environ.get("CSPRO_DEMO_SUPPORT_KEY"))
    ap.add_argument("--beta-key", default=os.environ.get("CSPRO_DEMO_BETA_KEY"))
    a = ap.parse_args()
    if not (a.alpha_key and a.support_key and a.beta_key):
        ap.error("the three demo keys are required (printed once by scripts.seed_demo)")

    with httpx.Client(base_url=a.base_url, timeout=120) as c:
        def ask(key, q):
            return c.post("/ask", json={"question": q}, headers={"X-API-Key": key})

        show("GET /health", c.get("/health"), keys=("status", "ledger", "model"))
        show("POST /ask without a key", c.post("/ask", json={"question": "x"}))
        show("POST /ask: cited answer", ask(a.alpha_key, "What is the refund window?"))
        show("POST /ask: question the documents cannot answer", ask(a.alpha_key, "Who won the 1998 football world cup?"))
        show("POST /ask: conflicting runbooks", ask(a.alpha_key, "What is the first response time for a P1 incident? Is there any conflicting information?"))

        stale = {"doc_id": "KB-001", "version": 1, "kind": "kb", "title": "Refund Policy",
                 "content": "Refund Policy (v1). Customers may request a refund within 14 days of delivery."}
        show("POST /ingest: Airflow retry replays OLD KB-001 v1 (v2 is already active)",
             c.post("/ingest", json=stale, headers={"X-API-Key": a.alpha_key, "X-Run-Id": "demo-run", "X-Dag-Id": "cspro_ingest",
                                                    "X-Task-Id": "post_manifest", "X-Try-Number": "2"}))
        d = c.get("/documents/KB-001", headers={"X-API-Key": a.alpha_key}).json()
        print(f"   still served: KB-001 version {d.get('version')} -> {d.get('content', '')[:80]!r}")

        r = ask(a.alpha_key, "What was ticket 1002 about? Is the parcel late?")
        body = show("POST /ask: ticket TCK-1002 contains a forged 'SYSTEM:' instruction", r)
        print(f"   beta canary in the response: {'CANARY-BETA' in r.text}   (must be False)")

        other = c.get("/documents/KB-101", headers={"X-API-Key": a.alpha_key})
        missing = c.get("/documents/KB-NOPE", headers={"X-API-Key": a.alpha_key})
        show("GET /documents/KB-101 (exists only in tenant beta)", other)
        print(f"   identical to a document that does not exist: "
              f"{other.status_code == missing.status_code and other.json()['error']['message'] == missing.json()['error']['message']}")
        show("GET /documents/KB-003 as alpha-support (restricted to group oncall)",
             c.get("/documents/KB-003", headers={"X-API-Key": a.support_key}))
        show("beta asks about alpha's refund policy", ask(a.beta_key, "What is the refund window of the Refund Policy v2?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
