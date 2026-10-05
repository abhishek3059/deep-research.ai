"""Abstract embedding provider interface.

Mirrors :mod:`src.vectorstore.base`: the vector store is reached through a
structural ``Protocol`` so backends are swappable, and the *same* pattern is
applied here to embedders (ADR-012).

Why this exists: :class:`~src.retrieval.pipeline.RetrievalPipeline` previously
type-hinted the concrete :class:`~src.ingestion.embedder.Embedder`, which meant
adding a second provider would have required editing every caller. With a
protocol, a new backend is a new class and nothing else.

Deliberately **structural** — implementations are not required to inherit. That
is why :class:`Embedder` already satisfies this protocol without modification:
``isinstance(Embedder(), EmbedderProtocol)`` is ``True`` today.

This satisfies contract section 4.1 (Ingestion -> VectorStore); embeddings are
the upstream half of that boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.config.settings import settings
from src.ingestion import EmbeddingError

if TYPE_CHECKING:
    from langchain_core.documents import Document


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Uniform interface every embedding provider must satisfy."""

    @property
    def model_name(self) -> str:
        """Name of the configured embedding model."""
        ...  # pragma: no cover

    async def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Embed documents, returning one vector per document in input order."""
        ...  # pragma: no cover

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings, returning one vector per input string in order."""
        ...  # pragma: no cover

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        ...  # pragma: no cover


def build_embedder(provider: str | None = None) -> EmbedderProtocol:
    """Construct the embedder named by configuration.

    The single point where provider selection happens. Callers depend on
    :class:`EmbedderProtocol`, so swapping ``EMBEDDING_PROVIDER`` in the
    environment changes the backend with no code change (ADR-012).

    Imports are deferred to keep ``langchain-google-genai`` off the import path
    for deployments that do not use it.

    Args:
        provider: Override for ``settings.embedding_provider``.

    Returns:
        An embedder satisfying :class:`EmbedderProtocol`.

    Raises:
        EmbeddingError: If the provider name is not recognised.
    """
    resolved = (provider or settings.embedding_provider).strip().lower()

    if resolved == "openai":
        from src.ingestion.embedder import Embedder

        return Embedder()

    if resolved == "gemini":
        from src.ingestion.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder()

    raise EmbeddingError(f"Unknown embedding provider {resolved!r}. Use 'openai' or 'gemini'.")
