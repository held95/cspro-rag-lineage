from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Response

from app.api.deps import get_container, get_run_context, require_scope
from app.container import Container
from app.core.errors import AppError
from app.modules.ingestion.ledger import IngestOutcome
from app.modules.tenancy.principal import Principal
from app.schemas.ingest import DocumentManifest, IngestResponse, RunContext

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(
    manifest: DocumentManifest,
    response: Response,
    principal: Principal = Depends(require_scope("docs:write")),
    run: RunContext = Depends(get_run_context),
    c: Container = Depends(get_container),
) -> IngestResponse:
    try:
        result = c.ingestion.ingest(principal.scope(), manifest, run)
    except (httpx.HTTPError, ConnectionError, OSError) as exc:
        raise AppError("DEPENDENCY_UNAVAILABLE", 503, "A required dependency is unavailable; retry later") from exc

    if result.outcome == IngestOutcome.REJECTED_STALE:
        raise AppError("STALE_VERSION", 409, "A newer version of this document was already accepted")
    if result.outcome == IngestOutcome.CONFLICT:
        raise AppError("VERSION_CONTENT_CONFLICT", 409, "This version already exists with different content")
    if result.outcome == IngestOutcome.NOOP_DUPLICATE:
        response.status_code = 200
    return IngestResponse(outcome=result.outcome.value, doc_id=result.doc_id, version=result.version,
                          active_version=result.active_version, status=result.status)
