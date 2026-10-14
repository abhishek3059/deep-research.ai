"""Health check endpoint (liveness + vector store readiness)."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    """Return service status, version, and vector store chunk count.

    503 when the store is unreachable — this is the readiness signal
    Render uses to decide whether the service can take traffic.
    """
    # Deferred import: building the store creates the ChromaDB client, which
    # must not happen at module import time (keeps tests hermetic).
    from src.vectorstore.manager import get_store

    try:
        store = get_store()
        # `count` is a backend extra outside the §4.2 protocol surface, so it
        # is probed structurally instead of widening the contract.
        count_attr = getattr(store, "count", None)
        chunks: int | None = await count_attr() if callable(count_attr) else None
    except Exception as exc:
        logger.error("Health check: vector store unreachable", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Vector store unreachable: {exc}") from exc
    return {
        "status": "ok",
        "version": "0.1.0",
        "vector_store": "chroma",
        "chunks": chunks,
    }
