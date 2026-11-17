from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.container import Container
from app.core.config import KeyRecord, Settings
from app.main import create_app
from app.modules.generation.service import AskService
from tests.fakes import ExtractiveFakeLLM
from tests.helpers import ingest_records, make_stack

KEYS = {
    "alpha-admin-key": KeyRecord(tenant_id="alpha", subject="alpha-admin", groups=["support", "oncall"],
                                 scopes=["docs:read", "docs:write", "ask:read"]),
    "alpha-support-key": KeyRecord(tenant_id="alpha", subject="alpha-support", groups=["support"],
                                   scopes=["docs:read", "ask:read"]),
    "beta-admin-key": KeyRecord(tenant_id="beta", subject="beta-admin", groups=[],
                                scopes=["docs:read", "docs:write", "ask:read"]),
}


def hashed_keys() -> dict[str, KeyRecord]:
    return {hashlib.sha256(k.encode()).hexdigest(): v for k, v in KEYS.items()}


@dataclass
class App:
    client: TestClient
    stack: object
    llm: object

    def h(self, key: str, **extra) -> dict:
        return {"X-API-Key": key, **extra}


def build_app(tmp_path, llm=None, stage: str | None = None, stack=None, retriever=None) -> App:
    stack = stack or make_stack(tmp_path)
    if retriever is not None:
        stack.retriever = retriever
    llm = llm or ExtractiveFakeLLM()
    settings = Settings(ledger_path=str(tmp_path / "ledger.db"), api_keys=hashed_keys(), llm_backend="fake",
                        embed_backend="hash", llm_model=getattr(llm, "name", "fake-llm"))
    container = Container(
        ledger=stack.ledger, store=stack.store, embedder=stack.embedder, ingestion=stack.service,
        retriever=stack.retriever, ask=AskService(stack.retriever, llm, getattr(llm, "name", "fake-llm")),
        llm=llm, embed_model=stack.embedder.name,
    )
    if stage:
        ingest_records(stack, stage)
    client = TestClient(create_app(settings, container), raise_server_exceptions=False)
    return App(client, stack, llm)


@pytest.fixture
def app_full(tmp_path):
    return build_app(tmp_path, stage="full")


@pytest.fixture
def app_lineage(tmp_path):
    return build_app(tmp_path, stage="lineage")
