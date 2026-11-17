"""One error shape for every non-2xx response:

    {"error": {"code": "...", "message": "...", "details": [...], "request_id": "..."}}

Validation details never echo the offending input and 500s never leak internals.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("app.errors")

_STATUS_CODES = {
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
}


class AppError(Exception):
    def __init__(self, code: str, status: int, message: str, details: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.code, self.status, self.message, self.details = code, status, message, details or []


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_body(code: str, message: str, request_id: str, details: list[dict[str, Any]] | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or [], "request_id": request_id}}


def error_response(code: str, status: int, message: str, request_id: str, details=None) -> JSONResponse:
    return JSONResponse(error_body(code, message, request_id, details), status_code=status,
                        headers={"X-Request-ID": request_id})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return error_response(exc.code, exc.status, exc.message, request_id_of(request), exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # keep loc/msg/type only: `input` and `ctx` could echo the payload or internals back
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return error_response("VALIDATION_ERROR", 422, "Request validation failed", request_id_of(request), details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return error_response(code, exc.status_code, message, request_id_of(request))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error("unhandled error request_id=%s type=%s", request_id_of(request), type(exc).__name__)
        return error_response("INTERNAL_ERROR", 500, "Internal server error", request_id_of(request))
