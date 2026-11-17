from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import ask, documents, health, ingest
from app.container import Container, build_container
from app.core.config import Settings
from app.core.errors import register_error_handlers
from app.core.middleware import RequestContextMiddleware


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.container is None:  # tests inject a Container built with fakes
            app.state.container = build_container(settings)
        yield

    app = FastAPI(title="CSPRO RAG lineage demo", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.container = container
    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_body_bytes)
    for module in (health, ingest, documents, ask):
        app.include_router(module.router)
    return app


def get_app() -> FastAPI:  # `uvicorn app.main:get_app --factory`
    return create_app()
