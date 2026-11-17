"""Shared builders for tests: wire a full in-process stack with fakes (no Ollama, no Docker)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.modules.ingestion.ledger import Ledger
from app.modules.ingestion.service import IngestionService
from app.modules.retrieval.embedder import HashEmbedder
from app.modules.retrieval.qdrant_store import QdrantStore
from app.modules.retrieval.retriever import Retriever
from app.modules.tenancy.principal import Principal
from app.schemas.ingest import DocumentManifest

RECORDS = json.loads((Path(__file__).parent.parent / "data" / "synthetic" / "records.json").read_text("utf-8"))


def principal(tenant: str, groups=(), scopes=("docs:read", "docs:write", "ask:read"), subject="tester") -> Principal:
    return Principal(tenant_id=tenant, subject=subject, groups=frozenset(groups), scopes=frozenset(scopes))


def manifest(doc_id="KB-001", version=1, content="Refund window is 14 days.", **kw) -> DocumentManifest:
    base = dict(doc_id=doc_id, version=version, kind="kb", title="Doc " + doc_id, content=content)
    base.update(kw)
    return DocumentManifest(**base)


@dataclass
class Stack:
    ledger: Ledger
    store: QdrantStore
    embedder: HashEmbedder
    service: IngestionService
    retriever: Retriever


def make_stack(tmp_path, min_score: float = 0.05, store=None) -> Stack:
    ledger = Ledger(str(tmp_path / "ledger.db"))
    embedder = HashEmbedder()
    store = store or QdrantStore(path=None, collection="chunks", dim=embedder.dim)
    service = IngestionService(ledger, store, embedder, embed_model=embedder.name)
    retriever = Retriever(ledger, store, embedder, min_score=min_score)
    return Stack(ledger, store, embedder, service, retriever)


def load_records(stage: str = "full"):
    return [r for r in RECORDS if stage in r["stages"]]


def ingest_records(stack: Stack, stage: str = "full"):
    for r in load_records(stage):
        m = DocumentManifest(
            doc_id=r["doc_id"], version=r["version"], kind=r["kind"], title=r["title"], content=r["content"],
            topic=r["topic"], visibility=r["visibility"], acl_groups=r["acl_groups"],
            effective_date=r["effective_date"],
        )
        stack.service.ingest(principal(r["tenant"]).scope(), m)
