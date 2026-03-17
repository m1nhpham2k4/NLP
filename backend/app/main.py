from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import bootstrap  # noqa: F401
from .api.routes import router
from .runtime import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(
        title="IUH Assistant API",
        summary="Separated backend for the IUH RAG assistant.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
