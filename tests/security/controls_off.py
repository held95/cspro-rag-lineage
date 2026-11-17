"""TEST-ONLY wrappers that switch a defence off. They exist so each "passes by construction"
test has a twin that proves the test CAN fail (SEC-24)."""
from __future__ import annotations

from app.modules.generation.verify import VerifiedAnswer
from app.modules.retrieval.retriever import Retriever
from app.modules.retrieval.store import ScoredChunk
from app.schemas.ask import Citation


class UnscopedStore:
    """Vector store that ignores the TenantScope (what a missing filter looks like)."""

    def __init__(self, inner):
        self.inner = inner

    def search(self, scope, vector, limit):
        res = self.inner.client.query_points(self.inner.collection, query=vector, limit=limit, with_payload=True)
        return [ScoredChunk(score=p.score, payload=p.payload) for p in res.points]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TrustingRetriever(Retriever):
    """Retriever without the independent ledger / tenant / ACL re-check."""

    def _authorized(self, principal, hit) -> bool:
        return True


def verifier_off(raw, ctx, question=""):
    """Replacement for verify_answer: believes whatever the model says."""
    cites = [
        Citation(source_id=c.source_id, doc_id="?", version=0, chunk_id="?", content_sha256="?",
                 source_uri=None, excerpt=c.quote)
        for c in raw.citations
    ]
    return VerifiedAnswer("answered", raw.answer, None, cites)
