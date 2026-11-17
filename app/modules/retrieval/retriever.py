from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.modules.ingestion.ledger import Ledger
from app.modules.retrieval.embedder import Embedder
from app.modules.retrieval.store import ScoredChunk, VectorStore
from app.modules.tenancy.authz import can_read
from app.modules.tenancy.principal import Principal

log = logging.getLogger("app.retrieval")


@dataclass
class AuthorizedContext:
    """Everything the LLM is allowed to see. Built only from hits that passed BOTH the in-query
    filter and the independent ledger re-check."""

    hits: list[ScoredChunk] = field(default_factory=list)

    @property
    def aliases(self) -> dict[str, ScoredChunk]:
        return {f"S{i}": h for i, h in enumerate(self.hits, start=1)}

    @property
    def conflict_doc_ids(self) -> list[str]:
        """Docs that share a topic with another retrieved doc (candidate conflicting evidence)."""
        by_topic: dict[str, set[str]] = {}
        for h in self.hits:
            if h.topic:
                by_topic.setdefault(h.topic, set()).add(h.doc_id)
        return sorted({d for docs in by_topic.values() if len(docs) > 1 for d in docs})


class Retriever:
    def __init__(self, ledger: Ledger, store: VectorStore, embedder: Embedder, min_score: float):
        self.ledger, self.store, self.embedder, self.min_score = ledger, store, embedder, min_score

    def retrieve(self, principal: Principal, query: str, top_k: int = 5) -> AuthorizedContext:
        scope = principal.scope()
        vector = self.embedder.embed([query], "query")[0]
        # over-fetch: stale/pending points that the ledger check drops must not eat the top-k slots
        raw = self.store.search(scope, vector, limit=top_k * 2)
        kept: list[ScoredChunk] = []
        for hit in raw:
            if hit.score < self.min_score:
                continue
            if not self._authorized(principal, hit):
                continue
            kept.append(hit)
            if len(kept) == top_k:
                break
        return AuthorizedContext(kept)

    def _authorized(self, principal: Principal, hit: ScoredChunk) -> bool:
        """Independent re-check against the ledger (source of truth), regardless of what the store returned."""
        scope = principal.scope()
        if hit.tenant_id != principal.tenant_id:
            log.error("security_drop: foreign-tenant chunk returned by the store")
            self.ledger.record_event(principal.tenant_id, hit.doc_id, hit.version, "security_drop", "foreign tenant chunk")
            return False
        row = self.ledger.get_version(principal.tenant_id, hit.doc_id, hit.version)
        if row is None or row.status != "active":
            return False  # pending / superseded versions are never served
        if self.ledger.get_active_version(principal.tenant_id, hit.doc_id) != hit.version:
            return False
        if not can_read(scope, principal.tenant_id, row.visibility, row.acl_groups):
            self.ledger.record_event(principal.tenant_id, hit.doc_id, hit.version, "security_drop", "acl")
            return False
        return True
