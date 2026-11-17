"""The BUGGY pipeline from the assessment scenario, kept on purpose as a reproducer.

It behaves like a typical "upsert by content" ingestion job:
  * no ledger, no version ordering - whatever payload arrives last becomes current
  * "replace" is implemented as delete-by-doc_id on a SHARED collection WITHOUT a tenant filter

An Airflow retry/clear that replays an old payload therefore republishes the superseded
document, and a v1->v2 supersede in one tenant wipes another tenant's document with the same id.
"""
from __future__ import annotations

import hashlib

from app.modules.generation.normalize import normalize


class NaivePipeline:
    def __init__(self):
        self.points: dict[str, dict] = {}  # the shared collection

    def ingest(self, tenant: str, doc_id: str, version: int, content: str) -> None:
        # BUG 1: delete-by-doc_id with no tenant filter on a shared collection
        for pid in [p for p, pl in self.points.items() if pl["doc_id"] == doc_id]:
            del self.points[pid]
        # BUG 2: no version check - the last write wins, even if it is an older version
        pid = hashlib.sha256(f"{tenant}|{doc_id}|{version}|{content}".encode()).hexdigest()
        self.points[pid] = {"tenant": tenant, "doc_id": doc_id, "version": version, "text": normalize(content).text}

    def visible_versions(self, tenant: str, doc_id: str) -> set[int]:
        return {pl["version"] for pl in self.points.values() if pl["tenant"] == tenant and pl["doc_id"] == doc_id}
