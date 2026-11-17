from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import health
from app.core.config import Settings
from app.core.errors import register_error_handlers
from app.core.middleware import RequestContextMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="CSPRO RAG lineage demo", version="0.1.0")
    app.state.settings = settings
    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)
    app.include_router(health.router)
    return app


def get_app() -> FastAPI:  # `uvicorn app.main:get_app --factory`
    return create_app()
