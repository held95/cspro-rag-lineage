from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container, require_scope
from app.container import Container
from app.core.errors import AppError
from app.modules.tenancy.authz import can_read
from app.modules.tenancy.principal import Principal

router = APIRouter(tags=["documents"])


def _not_found() -> AppError:
    # identical for "other tenant", "no such document" and "no permission": no existence oracle
    return AppError("NOT_FOUND", 404, "Document not found")


def _readable_versions(c: Container, principal: Principal, doc_id: str):
    rows = c.ledger.versions(principal.tenant_id, doc_id)  # ledger query is tenant-scoped
    if not rows or not can_read(principal.scope(), principal.tenant_id, rows[-1].visibility, rows[-1].acl_groups):
        raise _not_found()
    return rows


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, principal: Principal = Depends(require_scope("docs:read")),
                 c: Container = Depends(get_container)) -> dict:
    rows = _readable_versions(c, principal, doc_id)
    active = c.ledger.get_active_version(principal.tenant_id, doc_id)
    row = next((r for r in rows if r.version == active), None)
    if row is None:
        raise _not_found()
    return {"doc_id": row.doc_id, "version": row.version, "kind": row.kind, "title": row.title,
            "content": row.content, "content_sha256": row.content_sha256, "visibility": row.visibility}


@router.get("/documents/{doc_id}/lineage")
def get_lineage(doc_id: str, principal: Principal = Depends(require_scope("docs:read")),
                c: Container = Depends(get_container)) -> dict:
    rows = _readable_versions(c, principal, doc_id)
    return {
        "doc_id": doc_id,
        "active_version": c.ledger.get_active_version(principal.tenant_id, doc_id),
        "versions": [{"version": r.version, "status": r.status, "content_sha256": r.content_sha256,
                      "ingested_at": r.ingested_at, "activated_at": r.activated_at,
                      "run_id": r.run_id, "try_number": r.try_number} for r in rows],
        "events": c.ledger.events(principal.tenant_id, doc_id),
    }
