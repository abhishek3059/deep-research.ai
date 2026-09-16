"""Middleware for FastAPI application."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with method, path, status code, and duration."""

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
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


def setup_middleware(app: FastAPI) -> None:
    """Apply all middleware to the app."""
    add_cors_middleware(app)
    add_request_logging(app)