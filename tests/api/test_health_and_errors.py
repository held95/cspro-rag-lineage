import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def client(tmp_path):
    settings = Settings(ledger_path=str(tmp_path / "ledger.db"), max_body_bytes=1024)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        yield c


def test_api01_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_request_id_generated_and_echoed(client):
    r = client.get("/health")
    assert len(r.headers["x-request-id"]) >= 8
    r2 = client.get("/health", headers={"X-Request-ID": "abc-12345678"})
    assert r2.headers["x-request-id"] == "abc-12345678"


def test_invalid_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"})
    assert r.headers["x-request-id"] != "bad id\twith spaces"


def test_api10_404_uses_error_envelope(client):
    r = client.get("/nope")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "NOT_FOUND"
    assert err["request_id"] == r.headers["x-request-id"]


def test_api10_405_uses_error_envelope(client):
    r = client.post("/health")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_api10_500_hides_internals(tmp_path):
    settings = Settings(ledger_path=str(tmp_path / "l.db"))
    app = create_app(settings)

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail /etc/passwd")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    body = r.text
    assert "secret internal detail" not in body and "Traceback" not in body
    assert r.json()["error"]["code"] == "INTERNAL_ERROR"
    assert r.json()["error"]["request_id"]


def test_api04b_oversize_content_length_is_413(client):
    r = client.post("/health", content=b"x" * 2048)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_api04b_oversize_chunked_body_is_413(tmp_path):
    settings = Settings(ledger_path=str(tmp_path / "l.db"), max_body_bytes=1024)
    app = create_app(settings)

    @app.post("/echo")
    async def echo(request_body: dict):  # forces the body to be read
        return request_body

    def gen():  # no Content-Length -> chunked transfer
        for _ in range(10):
            yield b"x" * 512

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/echo", content=gen(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
