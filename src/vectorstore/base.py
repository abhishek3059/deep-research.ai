"""Abstract vector store protocol and shared data types.

Defines the :class:`VectorStoreProtocol` that every backend must satisfy,
plus :class:`SearchResult` used to return ranked results to the retrieval
layer.  These match the contracts in docs/CONTRACTS.md section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.ingestion.pipeline import ChunkMetadata


@dataclass
class SearchResult:
    """A single result returned by a vector store search."""

    id: str
    text: str
    metadata: ChunkMetadata
    score: float
    """Similarity score in [0, 1] — higher is better."""


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Uniform interface that all vector store backends must implement."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to ``query_embedding``."""
        ...  # pragma: no cover

    async def upsert(self, chunks: list[Any]) -> int:
        """Insert or update chunks, returning the count stored."""
        ...  # pragma: no cover

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by id, returning the count removed."""
        ...  # pragma: no cover
