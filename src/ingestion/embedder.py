"""Embedding generation using OpenAI embeddings.

All embedding I/O is async so the ingestion pipeline never blocks the
event loop. The model and API key come from :mod:`src.config.settings`;
nothing is hardcoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_openai import OpenAIEmbeddings

from src.config.settings import settings
from src.ingestion import EmbeddingError

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


class Embedder:
    """Generate dense vectors for documents and queries via OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        embeddings: OpenAIEmbeddings | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Configure the embedder.

        Args:
            model: Embedding model name. Defaults to ``settings.embedding_model``.
            embeddings: Pre-built embeddings client. When omitted, one is
                created from the application settings or the provided model.
            batch_size: Maximum number of texts sent per API request.

        Raises:
            EmbeddingError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be a positive integer")
        self._batch_size = batch_size
        self._embeddings = embeddings if embeddings is not None else self._build_default(model)

    @staticmethod
    def _build_default(model: str | None = None) -> OpenAIEmbeddings:
        """Build an OpenAI embeddings client from application settings."""
        resolved_model = model or settings.embedding_model
        if settings.openai_api_key:
            return OpenAIEmbeddings(
                model=resolved_model,
                openai_api_key=settings.openai_api_key,
            )
        return OpenAIEmbeddings(model=resolved_model)

    @property
    def model_name(self) -> str:
        """Name of the configured embedding model."""
        return str(self._embeddings.model)

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Embed documents in input order.

        Args:
            documents: Documents whose ``page_content`` should be embedded.

        Returns:
            One embedding vector per document, in the same order.

        Raises:
            EmbeddingError: If any embedding request fails.
        """
        return await self.embed_texts([document.page_content for document in documents])

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings, batching requests to bound payload size.

        Args:
            texts: Strings to embed.

        Returns:
            One embedding vector per input string, in the same order.

        Raises:
            EmbeddingError: If any embedding request fails.
        """
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        logger.info("Texts embedded", count=len(texts), batch_size=self._batch_size)
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query.

        Args:
            query: Natural language query string.

        Returns:
            The query embedding vector.

        Raises:
            EmbeddingError: If the embedding request fails.
        """
        try:
            return await self._embeddings.aembed_query(query)
        except Exception as exc:  # noqa: BLE001 - provider error types vary
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a single batch, translating provider errors."""
        try:
            return await self._embeddings.aembed_documents(batch)
        except Exception as exc:  # noqa: BLE001 - provider error types vary
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
