from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.modules.tenancy.principal import TenantScope
from app.schemas.ingest import DocumentManifest

_NS = uuid.UUID("6f1c5d0e-7a53-4d8e-9b1a-2c4f8e0a1b77")


@dataclass(frozen=True)
class ChunkPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]

    @classmethod
    def build(cls, tenant_id: str, m: DocumentManifest, ordinal: int, text: str, vector: list[float],
              embed_model: str) -> "ChunkPoint":
        chunk_sha = hashlib.sha256(text.encode()).hexdigest()
        # deterministic id: re-upserting the same chunk (a retry) overwrites instead of duplicating
        pid = str(uuid.uuid5(_NS, f"{tenant_id}:{m.doc_id}:{m.version}:{ordinal}:{chunk_sha[:16]}"))
        payload = {
            "tenant_id": tenant_id, "doc_id": m.doc_id, "version": m.version, "ordinal": ordinal,
            "chunk_id": pid, "chunk_text": text, "chunk_sha256": chunk_sha,
            "content_sha256": hashlib.sha256(m.content.encode()).hexdigest(),
            "kind": m.kind, "title": m.title, "topic": m.topic, "visibility": m.visibility,
            "acl_groups": list(m.acl_groups), "source_uri": m.source_uri,
            "effective_date": m.effective_date.isoformat() if m.effective_date else None,
            "embedding_model": embed_model,
        }
        return cls(pid, vector, payload)


@dataclass(frozen=True)
class ScoredChunk:
    score: float
    payload: dict[str, Any]

    tenant_id = property(lambda self: self.payload["tenant_id"])
    doc_id = property(lambda self: self.payload["doc_id"])
    version = property(lambda self: self.payload["version"])
    chunk_id = property(lambda self: self.payload["chunk_id"])
    text = property(lambda self: self.payload["chunk_text"])
    kind = property(lambda self: self.payload["kind"])
    title = property(lambda self: self.payload["title"])
    topic = property(lambda self: self.payload.get("topic"))
    visibility = property(lambda self: self.payload["visibility"])
    acl_groups = property(lambda self: self.payload["acl_groups"])
    content_sha256 = property(lambda self: self.payload["content_sha256"])
    source_uri = property(lambda self: self.payload.get("source_uri"))
    effective_date = property(lambda self: self.payload.get("effective_date"))


class VectorStore(Protocol):
    """Every method takes a TenantScope: there is no unscoped way to touch the shared collection."""

    def upsert(self, scope: TenantScope, points: list[ChunkPoint]) -> None: ...
    def search(self, scope: TenantScope, vector: list[float], limit: int) -> list[ScoredChunk]: ...
    def delete_versions(self, scope: TenantScope, doc_id: str, *, below: int | None = None,
                        only: int | None = None) -> None: ...
    def count(self, scope: TenantScope) -> int: ...
    def dump(self, scope: TenantScope) -> list[dict[str, Any]]: ...
