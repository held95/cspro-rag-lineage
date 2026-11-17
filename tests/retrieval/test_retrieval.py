"""RET: retrieval behaviour (top-k, score threshold, stale/superseded exclusion, ACL, tenant), using the hash embedder."""
import pytest

from tests.helpers import ingest_records, make_stack, manifest, principal

pytestmark = pytest.mark.unit

ALPHA = principal("alpha", groups=("support", "oncall"))
SUPPORT = principal("alpha", groups=("support",))


@pytest.fixture
def stack(tmp_path):
    s = make_stack(tmp_path)
    ingest_records(s, "full")
    return s


def test_ret01_relevant_document_is_ranked_first(stack):
    # The hash embedder is lexical: use distinctive vocabulary. Semantic quality is measured separately with the real
    # embedder (docs/RETRIEVAL_EVAL.md, scripts/eval_retrieval.py).
    hits = stack.retriever.retrieve(ALPHA, "Customers may request a refund within 30 days of delivery", 5).hits
    assert hits and hits[0].doc_id == "KB-001"
    assert hits[0].score >= max(h.score for h in hits)


def test_ret02_hits_are_sorted_by_score_and_top_k_is_respected(stack):
    for k in (1, 2, 3):
        hits = stack.retriever.retrieve(ALPHA, "refund delivery incident", k).hits
        assert len(hits) <= k
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_ret03_min_score_filters_weak_matches(tmp_path):
    loose = make_stack(tmp_path / "a", min_score=-1.0)
    strict = make_stack(tmp_path / "b", min_score=0.95)
    for s in (loose, strict):
        ingest_records(s, "full")
    q = "refund window"
    assert len(loose.retriever.retrieve(ALPHA, q, 5).hits) > len(strict.retriever.retrieve(ALPHA, q, 5).hits)
    assert strict.retriever.retrieve(ALPHA, "zzzz qqqq xxxx", 5).hits == []


def test_ret04_superseded_version_is_never_returned(stack):
    # KB-001 v1 (14 days) was superseded by v2 (30 days): only v2 may be served
    hits = stack.retriever.retrieve(ALPHA, "refund window days delivery", 10).hits
    kb1 = [h for h in hits if h.doc_id == "KB-001"]
    assert kb1 and {h.version for h in kb1} == {2}
    assert all("14 days" not in h.text for h in kb1)


def test_ret05_empty_index_returns_no_hits(tmp_path):
    s = make_stack(tmp_path)
    assert s.retriever.retrieve(ALPHA, "anything at all", 5).hits == []


def test_ret06_restricted_documents_need_the_group(stack):
    q = "How fast must a P1 incident get a first response?"
    support = {h.doc_id for h in stack.retriever.retrieve(SUPPORT, q, 10).hits}
    admin = {h.doc_id for h in stack.retriever.retrieve(ALPHA, q, 10).hits}
    assert not support & {"KB-003", "KB-004"}
    assert {"KB-003", "KB-004"} <= admin


def test_ret07_tenants_see_only_their_own_documents(stack):
    q = "returns accepted within how many days"
    assert {h.tenant_id for h in stack.retriever.retrieve(ALPHA, q, 10).hits} == {"alpha"}
    assert {h.tenant_id for h in stack.retriever.retrieve(principal("beta"), q, 10).hits} == {"beta"}


def test_ret08_conflict_candidates_are_flagged_by_shared_topic(stack):
    ctx = stack.retriever.retrieve(ALPHA, "P1 incident first response time escalation", 10)
    assert {"KB-003", "KB-004"} <= set(ctx.conflict_doc_ids)  # same topic p1_response, different claims


def test_ret09_aliases_map_one_to_one_to_hits(stack):
    ctx = stack.retriever.retrieve(ALPHA, "refund window", 3)
    assert list(ctx.aliases) == [f"S{i}" for i in range(1, len(ctx.hits) + 1)]
    assert [ctx.aliases[a] for a in ctx.aliases] == ctx.hits


def test_ret10_new_version_replaces_the_old_one_in_retrieval(tmp_path):
    s = make_stack(tmp_path)
    s.service.ingest(ALPHA.scope(), manifest("KB-050", 1, "Parking permits cost 10 euros per month."))
    s.service.ingest(ALPHA.scope(), manifest("KB-050", 2, "Parking permits cost 25 euros per month."))
    hits = s.retriever.retrieve(ALPHA, "parking permits cost per month", 5).hits
    assert [h.version for h in hits] == [2] and "25 euros" in hits[0].text
