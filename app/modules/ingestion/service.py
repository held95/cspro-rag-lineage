from __future__ import annotations

from dataclasses import dataclass

from app.modules.ingestion.chunking import chunk_document
from app.modules.ingestion.ledger import IngestOutcome, Ledger
from app.modules.retrieval.embedder import Embedder
from app.modules.retrieval.store import ChunkPoint, VectorStore
from app.modules.tenancy.principal import TenantScope
from app.schemas.ingest import DocumentManifest, RunContext

__all__ = ["IngestionService", "IngestOutcome", "IngestResult"]


@dataclass(frozen=True)
class IngestResult:
    outcome: IngestOutcome
    doc_id: str
    version: int
    active_version: int | None
    status: str  # status of the submitted version


class IngestionService:
    """accept (ledger) -> embed -> write vectors -> activate (ledger) -> delete older vectors.

    Safe to call repeatedly with the same payload (retries): stale versions are rejected, duplicates
    are no-ops, and a version that failed half-way is simply completed by the next attempt."""

    def __init__(self, ledger: Ledger, store: VectorStore, embedder: Embedder, embed_model: str):
        self.ledger, self.store, self.embedder, self.embed_model = ledger, store, embedder, embed_model

    def ingest(self, scope: TenantScope, m: DocumentManifest, run: RunContext | None = None) -> IngestResult:
        tenant = scope.tenant_id
        decision = self.ledger.accept(tenant, m, run, self.embed_model)
        if decision.needs_vectors:
            texts = chunk_document(m.content)
            vectors = self.embedder.embed(texts, "document")  # may raise: version stays 'pending', never served
            self.store.upsert(scope, [
                ChunkPoint.build(tenant, m, i, t, v, self.embed_model) for i, (t, v) in enumerate(zip(texts, vectors))
            ])
            if self.ledger.activate(tenant, m.doc_id, m.version):
                self.store.delete_versions(scope, m.doc_id, below=m.version)
            else:  # a newer version won while we were writing
                self.store.delete_versions(scope, m.doc_id, only=m.version)
        row = self.ledger.get_version(tenant, m.doc_id, m.version)
        return IngestResult(decision.outcome, m.doc_id, m.version, self.ledger.get_active_version(tenant, m.doc_id),
                            row.status if row else "unknown")
