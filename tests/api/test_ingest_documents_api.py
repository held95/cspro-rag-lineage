import pytest

from tests.conftest import build_app

pytestmark = pytest.mark.unit

BODY = {"doc_id": "KB-900", "version": 1, "kind": "kb", "title": "Test", "content": "Refund window is 14 days."}


@pytest.fixture
def app(tmp_path):
    return build_app(tmp_path)


def post(app, key, body, **headers):
    return app.client.post("/ingest", json=body, headers=app.h(key, **headers))


def test_api02_missing_and_invalid_key_are_401(app):
    assert app.client.post("/ingest", json=BODY).status_code == 401
    r = post(app, "wrong-key", BODY)
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_api09_missing_scope_is_403(app):
    r = post(app, "alpha-support-key", BODY)  # support key has no docs:write
    assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN_SCOPE"


def test_ingest_created_then_duplicate_is_200(app):
    assert post(app, "alpha-admin-key", BODY).status_code == 201
    r = post(app, "alpha-admin-key", BODY)
    assert r.status_code == 200 and r.json()["outcome"] == "noop_duplicate"


@pytest.mark.parametrize("patch", [
    {"tenant_id": "beta"},                      # API-05: forged tenant / unknown field
    {"unknown": 1},
    {"version": "1"},                           # API-06: StrictInt
    {"version": 0}, {"version": -1}, {"version": True},
    {"doc_id": "../x"}, {"doc_id": ""},
    {"content": "bad\x00nul"}, {"content": ""},
    {"kind": "email"},
    {"source_uri": "https://evil.example/x"},
])
def test_api05_06_invalid_payloads_are_422_without_echo(app, patch):
    r = post(app, "alpha-admin-key", {**BODY, **patch})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR" and err["request_id"]
    # details carry only loc/msg/type: no `input`/`ctx` keys, and the offending value is never echoed
    assert all(set(d) == {"loc", "msg", "type"} for d in err["details"])
    for bad in ("evil.example", "bad\x00nul", "../x"):
        assert bad not in r.text


def test_api07_invalid_utf8_body_is_422_or_400(app):
    r = app.client.post("/ingest", content=b'{"doc_id": "\xff\xfe"}',
                        headers={**app.h("alpha-admin-key"), "Content-Type": "application/json"})
    assert r.status_code in (400, 422)
    assert r.json()["error"]["request_id"]


def test_lin03_replayed_stale_version_is_409_and_lineage_shows_try_number(app):
    v1 = {**BODY, "doc_id": "KB-001", "version": 1}
    v2 = {**BODY, "doc_id": "KB-001", "version": 2, "content": "Refund window is 30 days."}
    run = {"X-Run-Id": "scheduled__2026-03-01", "X-Dag-Id": "ingest_dag", "X-Task-Id": "ingest_doc"}
    assert post(app, "alpha-admin-key", v1, **run, **{"X-Try-Number": "1"}).status_code == 201
    assert post(app, "alpha-admin-key", v2, **run, **{"X-Try-Number": "1"}).status_code == 201
    r = post(app, "alpha-admin-key", v1, **run, **{"X-Try-Number": "2"})  # the retry
    assert r.status_code == 409 and r.json()["error"]["code"] == "STALE_VERSION"
    lineage = app.client.get("/documents/KB-001/lineage", headers=app.h("alpha-admin-key")).json()
    stale = [e for e in lineage["events"] if e["action"] == "rejected_stale"]
    assert stale and stale[0]["try_number"] == 2 and stale[0]["run_id"] == "scheduled__2026-03-01"
    assert [v["status"] for v in lineage["versions"]] == ["superseded", "active"]


def test_same_version_different_content_is_409_conflict(app):
    post(app, "alpha-admin-key", BODY)
    r = post(app, "alpha-admin-key", {**BODY, "content": "Refund window is 99 days."})
    assert r.status_code == 409 and r.json()["error"]["code"] == "VERSION_CONTENT_CONFLICT"


def test_get_document_returns_active_version(app):
    post(app, "alpha-admin-key", BODY)
    r = app.client.get("/documents/KB-900", headers=app.h("alpha-admin-key"))
    assert r.status_code == 200 and r.json()["version"] == 1 and "14 days" in r.json()["content"]


def test_sec07_other_tenant_and_missing_documents_are_indistinguishable_404(app):
    post(app, "alpha-admin-key", BODY)
    other = app.client.get("/documents/KB-900", headers=app.h("beta-admin-key"))
    missing = app.client.get("/documents/NOPE", headers=app.h("beta-admin-key"))
    assert other.status_code == missing.status_code == 404

    def strip(r):
        body = r.json()
        body["error"].pop("request_id")
        return body, r.headers.get("content-type")

    assert strip(other) == strip(missing)


def test_sec23_restricted_document_and_lineage_are_404_without_group(tmp_path):
    app = build_app(tmp_path, stage="full")
    for path in ("/documents/KB-003", "/documents/KB-003/lineage"):
        assert app.client.get(path, headers=app.h("alpha-support-key")).status_code == 404
        assert app.client.get(path, headers=app.h("alpha-admin-key")).status_code == 200
        assert app.client.get(path, headers=app.h("beta-admin-key")).status_code == 404


def test_dependency_outage_during_ingest_is_503(tmp_path):
    from tests.ingestion.test_ingestion import FlakyEmbedder
    app = build_app(tmp_path)
    app.stack.service.embedder = FlakyEmbedder(app.stack.embedder)
    r = post(app, "alpha-admin-key", BODY)
    assert r.status_code == 503 and r.json()["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "embedder down" not in r.text and "ConnectionError" not in r.text  # no internal detail
