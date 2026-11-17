from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_container, require_scope
from app.container import Container
from app.core.errors import AppError, request_id_of
from app.modules.tenancy.principal import Principal
from app.schemas.ask import AskRequest, AskResponse

import httpx

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    principal: Principal = Depends(require_scope("ask:read")),
    c: Container = Depends(get_container),
) -> AskResponse:
    try:
        return c.ask.ask(principal, body, request_id_of(request))
    except AppError:
        raise
    except (httpx.HTTPError, ConnectionError, OSError) as exc:  # embedder or vector store down
        raise AppError("DEPENDENCY_UNAVAILABLE", 503, "A required dependency is unavailable; retry later") from exc
