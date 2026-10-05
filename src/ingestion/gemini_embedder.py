"""Embedding generation via Google Gemini.

The second implementation of :class:`~src.ingestion.embedder_base.EmbedderProtocol`
(ADR-012). Deliberately a near-mirror of
:class:`~src.ingestion.embedder.Embedder`: same batching, same error
translation, same public surface. The *only* meaningful difference is which
client is constructed.

Why Gemini needs its own class when chat does not: Google exposes an
**OpenAI-compatible chat endpoint** (:data:`GEMINI_OPENAI_BASE_URL`), so the LLM
provider reaches Gemini by swapping ``base_url`` on ``ChatOpenAI`` — a settings
change. That compatibility surface does **not** cover embeddings, so these go
through ``langchain-google-genai`` instead. The chat side is configuration;
the embedding side is a class. That asymmetry is the finding behind ADR-012.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config.settings import settings
from src.ingestion import EmbeddingError

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


class GeminiEmbedder:
    """Generate dense vectors via Google Gemini.

    Satisfies :class:`~src.ingestion.embedder_base.EmbedderProtocol`
    structurally — this class does not inherit from it, and does not need to.
    """

    def __init__(
        self,
        model: str | None = None,
        embeddings: GoogleGenerativeAIEmbeddings | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Configure the Gemini embedder.

        Args:
            model: Embedding model name. Defaults to
                ``settings.gemini_embedding_model``.
            embeddings: Pre-built client. When omitted, one is created from
                application settings.
            batch_size: Maximum number of texts per API request.

        Raises:
            EmbeddingError: If ``batch_size`` is not positive, or no API key is
                configured.
        """
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be a positive integer")
        self._batch_size = batch_size
        self._embeddings = embeddings if embeddings is not None else self._build_default(model)

    @staticmethod
    def _build_default(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
        """Build a Gemini embeddings client from application settings.

        Raises:
            EmbeddingError: If no Gemini API key is configured. Fail loudly
                rather than deferring the failure to the first embedding call.
        """
        if not settings.gemini_api_key:
            raise EmbeddingError(
                "GEMINI_API_KEY is not set. Add it to .env or switch "
                "EMBEDDING_PROVIDER back to 'openai'."
            )
        return GoogleGenerativeAIEmbeddings(
            model=model or settings.gemini_embedding_model,
            # `google_api_key` is a valid model field at runtime; mypy sees only
            # langchain-google-genai's plugin-style `__init__(self, data)`.
            google_api_key=settings.gemini_api_key,  # type: ignore[call-arg]
        )

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
            query: Search query string.

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
        """Embed a single batch, translating provider errors.

        Args:
            batch: Texts to embed in one request.

        Returns:
            One vector per text in the batch.

        Raises:
            EmbeddingError: If the request fails.
        """
        try:
            return await self._embeddings.aembed_documents(batch)
        except Exception as exc:  # noqa: BLE001 - provider error types vary
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc
