"""Pinecone vector store implementation (stub).

Placeholder for future Pinecone integration.  All methods raise
``NotImplementedError`` to make the unimplemented state explicit.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.ingestion.pipeline import ProcessedChunk
from src.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)


class PineconeStore:
    """Pinecone-backed vector store — not yet implemented."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search by vector similarity (stub)."""
        raise NotImplementedError("Pinecone search is not implemented")

    async def upsert(self, chunks: list[ProcessedChunk]) -> int:
        """Insert or update chunks (stub)."""
        raise NotImplementedError("Pinecone upsert is not implemented")

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by id (stub)."""
        raise NotImplementedError("Pinecone delete is not implemented")
