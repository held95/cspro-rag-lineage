from __future__ import annotations

import hashlib
import hmac
from typing import Callable

from fastapi import Depends, Header, Request

from app.container import Container
from app.core.config import Settings
from app.core.errors import AppError
from app.modules.tenancy.principal import Principal
from app.schemas.ingest import RunContext


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_principal(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    """The ONLY place a tenant is decided: API key -> Principal. Nothing in the body/prompt can change it."""
    settings: Settings = request.app.state.settings
    if not x_api_key:
        raise AppError("UNAUTHENTICATED", 401, "Missing API key")
    digest = hashlib.sha256(x_api_key.encode()).hexdigest()
    record = None
    for known, rec in settings.api_keys.items():  # constant-time compare, no early exit
        if hmac.compare_digest(known, digest):
            record = rec
    if record is None:
        raise AppError("UNAUTHENTICATED", 401, "Invalid API key")
    return Principal(record.tenant_id, record.subject, frozenset(record.groups), frozenset(record.scopes))


def require_scope(scope: str) -> Callable[..., Principal]:
    def dep(principal: Principal = Depends(get_principal)) -> Principal:
        if scope not in principal.scopes:
            raise AppError("FORBIDDEN_SCOPE", 403, f"Missing required scope: {scope}")
        return principal

    return dep


def get_run_context(
    x_run_id: str | None = Header(default=None, max_length=200),
    x_dag_id: str | None = Header(default=None, max_length=200),
    x_task_id: str | None = Header(default=None, max_length=200),
    x_try_number: int | None = Header(default=None, ge=0, le=1000),
    x_data_interval_start: str | None = Header(default=None, max_length=64),
) -> RunContext:
    """Orchestrator lineage metadata. Recorded for audit only - it never affects any decision."""
    return RunContext(x_run_id, x_dag_id, x_task_id, x_try_number, x_data_interval_start)
