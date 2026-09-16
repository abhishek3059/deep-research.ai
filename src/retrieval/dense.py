"""Dense vector retriever wrapping the VectorStoreProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.constants import DEFAULT_TOP_K
from src.vectorstore.base import SearchResult

if TYPE_CHECKING:
    from src.vectorstore.base import VectorStoreProtocol


class DenseRetriever:
    """Thin wrapper around a vector store for embedding-based retrieval."""

    def __init__(self, vector_store: VectorStoreProtocol) -> None:
        self._vector_store = vector_store

    async def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Return the top_k nearest neighbours for *query_embedding*.

        Args:
            query_embedding: Dense embedding of the query.
            top_k: Maximum results to return.

        Returns:
            Ranked list of SearchResult.
        """
        return await self._vector_store.search(query_embedding, top_k=top_k)
