import itertools
import random

import pytest

from app.modules.ingestion.service import IngestOutcome
from tests.helpers import make_stack, manifest, principal

pytestmark = pytest.mark.unit

ALPHA = principal("alpha")
BETA = principal("beta")


def ingest(stack, who, doc_id="KB-001", version=1, content="Refund window is 14 days.", run=None, **kw):
    return stack.service.ingest(who.scope(), manifest(doc_id, version, content, **kw), run)


def count(stack, who):
    return stack.store.count(who.scope())


def test_ing01_first_ingest_activates_and_writes_points(tmp_path):
    s = make_stack(tmp_path)
    r = ingest(s, ALPHA)
    assert r.outcome == IngestOutcome.ACCEPTED and r.active_version == 1
    assert s.ledger.get_active_version("alpha", "KB-001") == 1
    assert count(s, ALPHA) >= 1


def test_ing02_duplicate_is_noop(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA)
    before = count(s, ALPHA)
    r = ingest(s, ALPHA)
    assert r.outcome == IngestOutcome.NOOP_DUPLICATE
    assert count(s, ALPHA) == before


def test_ing03_supersede_deletes_old_points_and_activates_new(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=1, content="Refund window is 14 days.")
    r = ingest(s, ALPHA, version=2, content="Refund window is 30 days.")
    assert r.active_version == 2
    versions = {p["version"] for p in s.store.dump(ALPHA.scope())}
    assert versions == {2}
    assert s.ledger.get_version("alpha", "KB-001", 1).status == "superseded"


def test_ing04_out_of_order_older_version_is_rejected(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=2, content="Refund window is 30 days.")
    r = ingest(s, ALPHA, version=1, content="Refund window is 14 days.")
    assert r.outcome == IngestOutcome.REJECTED_STALE
    assert s.ledger.get_active_version("alpha", "KB-001") == 2


def test_ing05_any_order_and_duplicates_end_with_highest_version_active(tmp_path):
    contents = {1: "refund 14 days", 2: "refund 30 days", 3: "refund 45 days"}
    orders = sorted(set(itertools.permutations([1, 2, 3, 1, 2, 3, 2])))
    orders = random.Random(0).sample(orders, 40)  # seeded sample of delivery orders incl. duplicates
    for i, order in enumerate(orders):
        s = make_stack(tmp_path / f"run{i}")
        for v in order:
            ingest(s, ALPHA, version=v, content=contents[v])
        assert s.ledger.get_active_version("alpha", "KB-001") == 3
        assert {p["version"] for p in s.store.dump(ALPHA.scope())} == {3}
        rows = s.ledger.versions("alpha", "KB-001")  # versions that arrived after v3 were never stored
        assert [r.version for r in rows if r.status == "active"] == [3]


def test_ing06_same_version_different_content_is_conflict(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=2, content="Refund window is 30 days.")
    r = ingest(s, ALPHA, version=2, content="Refund window is 99 days.")
    assert r.outcome == IngestOutcome.CONFLICT
    assert "30 days" in s.ledger.get_version("alpha", "KB-001", 2).content


def test_ing12_same_doc_id_in_two_tenants_is_independent(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=3, content="alpha text refund")
    r = ingest(s, BETA, version=1, content="beta text refund")  # v1 is NOT stale for beta
    assert r.outcome == IngestOutcome.ACCEPTED
    assert s.ledger.get_active_version("alpha", "KB-001") == 3
    assert s.ledger.get_active_version("beta", "KB-001") == 1


# ---- failure injection ------------------------------------------------------------------------

class FlakyStore:
    """Wraps a store; fails the n-th upsert call (simulates Qdrant down / worker crash)."""

    def __init__(self, inner, fail_on_call=1):
        self.inner, self.fail_on_call, self.calls = inner, fail_on_call, 0

    def upsert(self, scope, points):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise ConnectionError("vector store unavailable")
        return self.inner.upsert(scope, points)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class FlakyEmbedder:
    def __init__(self, inner):
        self.inner, self.fail = inner, True
        self.dim, self.name = inner.dim, inner.name

    def embed(self, texts, kind="document"):
        if self.fail:
            raise ConnectionError("embedder down")
        return self.inner.embed(texts, kind)


def test_ing08_crash_before_vectors_keeps_previous_version_served_and_retry_completes(tmp_path):
    base = make_stack(tmp_path)
    ingest(base, ALPHA, version=1, content="Refund window is 14 days.")
    flaky = FlakyStore(base.store, fail_on_call=1)
    from app.modules.ingestion.service import IngestionService
    svc = IngestionService(base.ledger, flaky, base.embedder, embed_model=base.embedder.name)

    with pytest.raises(ConnectionError):
        svc.ingest(ALPHA.scope(), manifest("KB-001", 2, "Refund window is 30 days."))

    # v2 is pending and NOT served; v1 is still the active, retrievable version
    assert base.ledger.get_version("alpha", "KB-001", 2).status == "pending"
    assert base.ledger.get_active_version("alpha", "KB-001") == 1
    ctx = base.retriever.retrieve(ALPHA, "refund window days", top_k=5)
    assert {h.version for h in ctx.hits} == {1}

    # ING-13: replaying v1 is stale and changes nothing; re-sending v2 completes it
    assert ingest(base, ALPHA, version=1, content="Refund window is 14 days.").outcome == IngestOutcome.REJECTED_STALE
    r = svc.ingest(ALPHA.scope(), manifest("KB-001", 2, "Refund window is 30 days."))
    assert r.outcome == IngestOutcome.NOOP_DUPLICATE and r.active_version == 2
    assert {p["version"] for p in base.store.dump(ALPHA.scope())} == {2}


def test_ing10_embedder_failure_leaves_previous_version_served(tmp_path):
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=1, content="Refund window is 14 days.")
    from app.modules.ingestion.service import IngestionService
    emb = FlakyEmbedder(s.embedder)
    svc = IngestionService(s.ledger, s.store, emb, embed_model=emb.name)
    with pytest.raises(ConnectionError):
        svc.ingest(ALPHA.scope(), manifest("KB-001", 2, "Refund window is 30 days."))
    assert s.ledger.get_active_version("alpha", "KB-001") == 1
    ctx = s.retriever.retrieve(ALPHA, "refund window days", top_k=5)
    assert ctx.hits and {h.version for h in ctx.hits} == {1}


def test_ing09_crash_after_vectors_before_activation_never_serves_pending(tmp_path):
    """Vectors for v2 exist but the ledger never activated it: retrieval must keep serving v1."""
    s = make_stack(tmp_path)
    ingest(s, ALPHA, version=1, content="Refund window is 14 days.")
    s.ledger.accept("alpha", manifest("KB-001", 2, "Refund window is 30 days."), None, s.embedder.name)
    from app.modules.ingestion.chunking import chunk_document
    from app.modules.retrieval.store import ChunkPoint
    texts = chunk_document("Refund window is 30 days.")
    vecs = s.embedder.embed(texts)
    s.store.upsert(ALPHA.scope(), [
        ChunkPoint.build("alpha", manifest("KB-001", 2, "Refund window is 30 days."), i, t, v, s.embedder.name)
        for i, (t, v) in enumerate(zip(texts, vecs))
    ])
    ctx = s.retriever.retrieve(ALPHA, "refund window days", top_k=5)
    assert {h.version for h in ctx.hits} == {1}          # pending v2 is dropped by the ledger re-check
    # the retry (same payload) completes activation and cleans v1 up
    r = ingest(s, ALPHA, version=2, content="Refund window is 30 days.")
    assert r.active_version == 2
    assert {p["version"] for p in s.store.dump(ALPHA.scope())} == {2}
