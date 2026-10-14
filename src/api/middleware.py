"""Middleware for FastAPI application."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        started = time.perf_counter()
        # Starlette types `call_next` as returning `Any`; the ASGI contract
        # guarantees a Response here.
        response = cast(Response, await call_next(request))
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 2),
        )
        return response


def add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware with origins from application settings."""
    from fastapi.middleware.cors import CORSMiddleware

    from src.config.settings import settings

    origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def add_request_logging(app: FastAPI) -> None:
    """Add request logging middleware."""
    app.add_middleware(RequestLoggingMiddleware)


def add_exception_handlers(app: FastAPI) -> None:
    """Map domain errors to structured JSON responses (stable error shape)."""
    from src.agents.exceptions import AgentError
    from src.ingestion import EmbeddingError, IngestionError
    from src.vectorstore import VectorStoreConnectionError, VectorStoreError

    @app.exception_handler(IngestionError)
    async def _ingestion_handler(_: Request, exc: IngestionError) -> JSONResponse:
        status = 502 if isinstance(exc, EmbeddingError) else 400
        logger.warning("Ingestion error", error=str(exc), status_code=status)
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.exception_handler(VectorStoreError)
    async def _vectorstore_handler(_: Request, exc: VectorStoreError) -> JSONResponse:
        status = 503 if isinstance(exc, VectorStoreConnectionError) else 500
        logger.error("Vector store error", error=str(exc), status_code=status)
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.exception_handler(AgentError)
    async def _agent_handler(_: Request, exc: AgentError) -> JSONResponse:
        logger.error("Agent error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )


def setup_middleware(app: FastAPI) -> None:
    """Apply all middleware to the app."""
    add_cors_middleware(app)
    add_request_logging(app)
    add_exception_handlers(app)
