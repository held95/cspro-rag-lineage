"""Generation: verify_answer, AskService and POST /ask (CIT, RAG, FAIL)."""
from __future__ import annotations

import pytest

from app.modules.generation.verify import verify_answer
from app.modules.retrieval.retriever import AuthorizedContext
from app.modules.retrieval.store import ScoredChunk
from app.schemas.ask import ABSTAIN_TEXT, GroundedAnswer, LLMCitation
from tests.conftest import build_app
from tests.fakes import ExtractiveFakeLLM, FlakyLLM, MalformedLLM, SlowLLM, WrongNumberLLM

pytestmark = pytest.mark.unit

ADMIN, SUPPORT, BETA = "alpha-admin-key", "alpha-support-key", "beta-admin-key"
REFUND_Q = "How many days do customers have to request a refund?"
P1_Q = "How quickly must a P1 incident receive a first response?"
TEXT_A = "Customers may request a refund within 30 days of delivery. Refunds take 5 business days."
TEXT_B = "A P1 incident must receive a first response within 60 minutes."


def chunk(doc_id: str, text: str, version: int = 1, topic: str | None = None) -> ScoredChunk:
    return ScoredChunk(1.0, {
        "tenant_id": "alpha", "doc_id": doc_id, "version": version, "chunk_id": f"{doc_id}:v{version}:0",
        "chunk_text": text, "kind": "kb", "title": "T " + doc_id, "topic": topic, "visibility": "tenant",
        "acl_groups": [], "content_sha256": "ab" * 32, "source_uri": None,
    })


def ctx(*chunks: ScoredChunk) -> AuthorizedContext:
    return AuthorizedContext(hits=list(chunks))


def ga(answer="", citations=(), abstain=False, conflict=False) -> GroundedAnswer:
    return GroundedAnswer(answer=answer, abstain=abstain, conflict=conflict,
                          citations=[LLMCitation(source_id=s, quote=q) for s, q in citations])


def ask(app, key, question=REFUND_Q, **extra):
    return app.client.post("/ask", json={"question": question, **extra}, headers=app.h(key))


# ----------------------------------------------------------------------------- CIT (verify_answer)

def test_cit01_verbatim_quote_accepted():
    v = verify_answer(ga("Refunds are allowed within 30 days.", [("S1", "within 30 days of delivery")]),
                      ctx(chunk("KB-001", TEXT_A)))
    assert v.status == "answered" and v.reason is None
    assert len(v.citations) == 1
    c = v.citations[0]
    assert (c.doc_id, c.version, c.chunk_id) == ("KB-001", 1, "KB-001:v1:0")
    assert c.content_sha256 == "ab" * 32
    assert "30 days" in c.excerpt


def test_cit02_nonexistent_alias_rejected():
    v = verify_answer(ga("Refund is 30 days.", [("S42", "30 days")]), ctx(chunk("KB-001", TEXT_A)))
    assert (v.status, v.reason, v.answer) == ("abstained", "unverifiable_output", ABSTAIN_TEXT)
    assert v.citations == []


def test_cit03_invented_quote_rejected():
    v = verify_answer(ga("Refund is 30 days.", [("S1", "this sentence is not in the source")]),
                      ctx(chunk("KB-001", TEXT_A)))
    assert (v.status, v.reason) == ("abstained", "unverifiable_output")


def test_cit04_no_citation_is_abstained_unverifiable():
    v = verify_answer(ga("Refund is 30 days.", []), ctx(chunk("KB-001", TEXT_A)))
    assert (v.status, v.reason) == ("abstained", "unverifiable_output")


def test_cit05_duplicate_citation_not_duplicated():
    q = "within 30 days of delivery"
    v = verify_answer(ga("Refunds are allowed within 30 days.", [("S1", q), ("S1", q)]),
                      ctx(chunk("KB-001", TEXT_A)))
    assert v.status == "answered"
    assert len(v.citations) == 1


def test_cit06_valid_citation_survives_alongside_forged_one():
    v = verify_answer(ga("Refunds are allowed within 30 days.", [("S9", "x"), ("S1", "within 30 days")]),
                      ctx(chunk("KB-001", TEXT_A)))
    assert v.status == "answered" and [c.source_id for c in v.citations] == ["S1"]


def test_cit07_model_abstain_flag_is_insufficient_evidence():
    v = verify_answer(ga(abstain=True), ctx(chunk("KB-001", TEXT_A)))
    assert (v.status, v.reason, v.answer) == ("abstained", "insufficient_evidence", ABSTAIN_TEXT)


def test_cit08_number_from_question_is_allowed():
    v = verify_answer(ga("Yes, 30 days applies.", [("S1", "refund within 30 days")]),
                      ctx(chunk("KB-001", TEXT_A)), question="Is 30 days the window?")
    assert v.status == "answered"


def test_cit09_conflict_flag_with_single_doc_is_plain_answer():
    v = verify_answer(ga("Within 30 days.", [("S1", "within 30 days")], conflict=True),
                      ctx(chunk("KB-001", TEXT_A)))
    assert v.status == "answered" and v.conflict_doc_ids == []


def test_cit10_conflict_two_docs_reports_sorted_doc_ids():
    v = verify_answer(
        ga("15 minutes vs 60 minutes.", [("S1", "within 15 minutes"), ("S2", "within 60 minutes")], conflict=True),
        ctx(chunk("KB-003", "A P1 incident must receive a first response within 15 minutes."),
            chunk("KB-004", TEXT_B)))
    assert v.status == "conflict" and v.conflict_doc_ids == ["KB-003", "KB-004"]


# ----------------------------------------------------------------------------- RAG (service + route)

def test_rag01_answer_with_evidence_has_full_citations(tmp_path):
    app = build_app(tmp_path, stage="full")
    r = ask(app, ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["reason"] is None
    assert "30 days" in body["answer"]
    assert body["conflict"] == {"detected": False, "doc_ids": []}
    assert body["citations"]
    c = body["citations"][0]
    for k in ("doc_id", "version", "chunk_id", "content_sha256", "excerpt"):
        assert c[k] not in (None, "")
    assert (c["doc_id"], c["version"]) == ("KB-001", 2)
    assert body["request_id"] == r.headers["x-request-id"]
    assert len(app.llm.calls) == 1


def test_rag02_irrelevant_question_abstains(tmp_path):
    app = build_app(tmp_path, stage="full")
    r = ask(app, ADMIN, "zzzz qqqq xxxx")
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["reason"], body["answer"]) == ("abstained", "insufficient_evidence", ABSTAIN_TEXT)
    assert body["citations"] == []


def test_rag02b_empty_index_abstains_without_calling_llm(tmp_path):
    app = build_app(tmp_path)  # nothing ingested -> no hits
    body = ask(app, ADMIN).json()
    assert (body["status"], body["reason"], body["answer"]) == ("abstained", "insufficient_evidence", ABSTAIN_TEXT)
    assert app.llm.calls == []


def test_rag03_invented_number_abstains(tmp_path):
    app = build_app(tmp_path, llm=WrongNumberLLM(), stage="full")
    body = ask(app, ADMIN).json()
    assert (body["status"], body["reason"], body["answer"]) == ("abstained", "unverifiable_output", ABSTAIN_TEXT)
    assert body["citations"] == []


def test_rag04_conflict_detected_with_two_cited_docs(tmp_path):
    app = build_app(tmp_path, stage="full")
    r = ask(app, ADMIN, P1_Q)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "conflict"
    assert body["conflict"]["detected"] is True
    assert body["conflict"]["doc_ids"] == ["KB-003", "KB-004"]
    assert {c["doc_id"] for c in body["citations"]} == {"KB-003", "KB-004"}


def test_rag05_restricted_conflict_docs_not_visible_without_group(tmp_path):
    app = build_app(tmp_path, stage="full")
    body = ask(app, SUPPORT, P1_Q).json()
    assert body["status"] == "abstained"
    assert body["conflict"]["doc_ids"] == []


def test_rag06_response_reports_model_name(tmp_path):
    app = build_app(tmp_path, stage="full")
    assert ask(app, ADMIN).json()["model"] == "extractive-fake"


# ----------------------------------------------------------------------------- CIT via the full pipeline

@pytest.mark.parametrize("kind", ["fake_alias", "fake_quote", "no_citation"])
def test_cit11_pipeline_unverifiable_llm_output_abstains(tmp_path, kind):
    llm = MalformedLLM(kind)
    app = build_app(tmp_path, llm=llm, stage="full")
    r = ask(app, ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["reason"], body["answer"]) == ("abstained", "unverifiable_output", ABSTAIN_TEXT)
    assert body["citations"] == []
    assert llm.calls == 1  # valid JSON -> no retry


# ----------------------------------------------------------------------------- FAIL

def test_fail01_malformed_json_retried_once_then_abstains(tmp_path):
    llm = MalformedLLM("bad_json")
    app = build_app(tmp_path, llm=llm, stage="full")
    r = ask(app, ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert (body["status"], body["reason"], body["answer"]) == ("abstained", "unverifiable_output", ABSTAIN_TEXT)
    assert llm.calls == 2


def test_fail02_llm_down_returns_503_envelope(tmp_path):
    llm = FlakyLLM(fail_times=5)
    app = build_app(tmp_path, llm=llm, stage="full")
    r = ask(app, ADMIN)
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "LLM_UNAVAILABLE"
    assert err["request_id"] == r.headers["x-request-id"]
    assert llm.calls == 1  # transport errors are not retried


def test_fail02b_flaky_llm_recovers_on_next_request(tmp_path):
    app = build_app(tmp_path, llm=FlakyLLM(fail_times=1), stage="full")
    assert ask(app, ADMIN).status_code == 503
    assert ask(app, ADMIN).status_code == 200


def test_fail03_llm_timeout_returns_504_envelope(tmp_path):
    app = build_app(tmp_path, llm=SlowLLM(), stage="full")
    r = ask(app, ADMIN)
    assert r.status_code == 504
    err = r.json()["error"]
    assert err["code"] == "LLM_TIMEOUT"
    assert err["request_id"] == r.headers["x-request-id"]


def test_fail04_error_envelope_echoes_client_request_id(tmp_path):
    app = build_app(tmp_path, llm=SlowLLM(), stage="full")
    r = app.client.post("/ask", json={"question": REFUND_Q}, headers=app.h(ADMIN, **{"X-Request-ID": "req-abc-12345"}))
    assert r.json()["error"]["request_id"] == "req-abc-12345"


@pytest.mark.parametrize("top_k", [0, -1, 11, "abc", 1.5])
def test_fail05_invalid_top_k_is_422(tmp_path, top_k):
    app = build_app(tmp_path, stage="full")
    r = ask(app, ADMIN, top_k=top_k)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR" and err["request_id"]
    assert app.llm.calls == []


@pytest.mark.parametrize("q", ["", "   ", "\n\t"])
def test_fail06_empty_question_is_422(tmp_path, q):
    app = build_app(tmp_path, stage="full")
    r = ask(app, ADMIN, q)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_fail06b_overlong_question_and_extra_field_are_422(tmp_path):
    app = build_app(tmp_path, stage="full")
    assert ask(app, ADMIN, "a" * 1001).status_code == 422
    assert ask(app, ADMIN, bogus=1).status_code == 422


def test_fail07_missing_key_is_401(tmp_path):
    app = build_app(tmp_path, stage="full")
    r = app.client.post("/ask", json={"question": REFUND_Q})
    assert r.status_code == 401
    err = r.json()["error"]
    assert err["code"] == "UNAUTHENTICATED" and err["request_id"]
    assert app.llm.calls == []


def test_fail08_invalid_key_is_401(tmp_path):
    app = build_app(tmp_path, stage="full")
    r = ask(app, "not-a-real-key")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_fail09_tenant_scoped_answer(tmp_path):
    app = build_app(tmp_path, stage="full")
    body = ask(app, BETA, "Returns accepted within how many days of purchase?").json()
    assert body["status"] == "answered"
    assert all(c["doc_id"] != "KB-003" for c in body["citations"])
    assert "7 days" in body["answer"]
