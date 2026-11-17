"""Qdrant-backed vector store: ONE shared collection, tenants separated by payload.

Every operation builds its filter from the caller's TenantScope; the tenant condition is part of
the ANN query itself (not a post-filter on the top-k)."""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.modules.retrieval.store import ChunkPoint, ScoredChunk
from app.modules.tenancy.principal import TenantScope


class QdrantStore:
    def __init__(self, path: str | None, collection: str, dim: int, url: str | None = None, api_key: str | None = None):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            try:  # keyword index flagged as the tenant key (server mode; harmless no-op locally)
                self.client.create_payload_index(
                    collection, "tenant_id", models.KeywordIndexParams(type="keyword", is_tenant=True)
                )
            except Exception:  # noqa: BLE001 - local mode does not need/support it
                pass

    # ---- filters -------------------------------------------------------------------------
    @staticmethod
    def _tenant_filter(scope: TenantScope, extra: list | None = None) -> models.Filter:
        must: list = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=scope.tenant_id))]
        must.extend(extra or [])
        return models.Filter(must=must)

    @staticmethod
    def _acl_condition(scope: TenantScope) -> models.Filter:
        """visibility == tenant  OR  doc ACL shares a group with the caller."""
        should: list = [models.FieldCondition(key="visibility", match=models.MatchValue(value="tenant"))]
        if scope.groups:
            should.append(models.FieldCondition(key="acl_groups", match=models.MatchAny(any=sorted(scope.groups))))
        return models.Filter(should=should)

    # ---- operations ----------------------------------------------------------------------
    def upsert(self, scope: TenantScope, points: list[ChunkPoint]) -> None:
        for p in points:
            if p.payload["tenant_id"] != scope.tenant_id:
                raise PermissionError("refusing to write a point for a different tenant")
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            wait=True,
        )

    def search(self, scope: TenantScope, vector: list[float], limit: int) -> list[ScoredChunk]:
        flt = self._tenant_filter(scope, [self._acl_condition(scope)])
        res = self.client.query_points(
            self.collection, query=vector, query_filter=flt, limit=limit, with_payload=True
        )
        return [ScoredChunk(score=p.score, payload=p.payload) for p in res.points]

    def delete_versions(self, scope: TenantScope, doc_id: str, *, below: int | None = None,
                        only: int | None = None) -> None:
        extra: list = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        if below is not None:
            extra.append(models.FieldCondition(key="version", range=models.Range(lt=below)))
        if only is not None:
            extra.append(models.FieldCondition(key="version", match=models.MatchValue(value=only)))
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._tenant_filter(scope, extra)),
            wait=True,
        )

    def count(self, scope: TenantScope) -> int:
        return self.client.count(self.collection, count_filter=self._tenant_filter(scope), exact=True).count

    def dump(self, scope: TenantScope) -> list[dict]:
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(
                self.collection, scroll_filter=self._tenant_filter(scope), limit=100,
                offset=offset, with_payload=True, with_vectors=False,
            )
            out.extend(p.payload for p in pts)
            if offset is None:
                return out
