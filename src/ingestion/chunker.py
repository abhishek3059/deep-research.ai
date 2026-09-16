"""Document chunking strategies for the ingestion pipeline.

Supports recursive character splitting (default) and semantic
embedding-based splitting. Recursive splitting is pure CPU work, while the
semantic splitter needs an embeddings instance to score sentence boundaries.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

from src.config.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkStrategy,
)
from src.ingestion import ChunkingError

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

logger = structlog.get_logger(__name__)


class TextChunker:
    """Split documents into smaller pieces using a chosen strategy."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Configure the chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters shared between adjacent chunks.

        Raises:
            ChunkingError: If the size/overlap combination is invalid.
        """
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_recursive(self, documents: list[Document]) -> list[Document]:
        """Split documents using recursive character splitting.

        Args:
            documents: Documents to split.

        Returns:
            Chunk documents, preserving source metadata and input order.

        Raises:
            ChunkingError: If the splitter fails.
        """
        return self._split(documents, ChunkStrategy.RECURSIVE)

    def chunk_semantic(
        self,
        documents: list[Document],
        embeddings: Embeddings,
    ) -> list[Document]:
        """Split documents using semantic embedding-based splitting.

        Args:
            documents: Documents to split.
            embeddings: Embeddings instance for computing sentence boundaries.

        Returns:
            Chunk documents, preserving source metadata and input order.

        Raises:
            ChunkingError: If the splitter fails or embeddings is None.
        """
        if embeddings is None:
            raise ChunkingError("Semantic chunking requires an embeddings instance")
        return self._split(documents, ChunkStrategy.SEMANTIC, embeddings)

    def chunk(
        self,
        documents: list[Document],
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        embeddings: Embeddings | None = None,
    ) -> list[Document]:
        """Split documents using the specified strategy.

        Factory method that dispatches to the appropriate chunking strategy.

        Args:
            documents: Documents to split.
            strategy: Chunking strategy to apply.
            embeddings: Required only for the semantic strategy.

        Returns:
            Chunk documents, preserving source metadata and input order.

        Raises:
            ChunkingError: If the strategy is unsupported or parameters are invalid.
        """
        if strategy is ChunkStrategy.RECURSIVE:
            return self.chunk_recursive(documents)
        if strategy is ChunkStrategy.SEMANTIC:
            return self.chunk_semantic(documents, embeddings)  # type: ignore[arg-type]
        raise ChunkingError(f"Unsupported chunk strategy: {strategy!r}")

    def _split(
        self,
        documents: list[Document],
        strategy: ChunkStrategy,
        embeddings: Embeddings | None = None,
    ) -> list[Document]:
        """Internal: build a splitter and split documents."""
        if not documents:
            return []

        splitter = self._build_splitter(strategy, embeddings)
        try:
            chunks = splitter.split_documents(documents)
        except ChunkingError:
            raise
        except Exception as exc:  # noqa: BLE001 - splitter error types vary
            raise ChunkingError(f"Failed to split documents: {exc}") from exc

        filtered = [chunk for chunk in chunks if chunk.page_content.strip()]
        logger.info(
            "Documents chunked",
            strategy=str(strategy),
            input_documents=len(documents),
            output_chunks=len(filtered),
        )
        return filtered

    def _build_splitter(
        self,
        strategy: ChunkStrategy,
        embeddings: Embeddings | None = None,
    ) -> TextSplitter:
        """Construct the underlying LangChain text splitter."""
        if strategy is ChunkStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        if strategy is ChunkStrategy.SEMANTIC:
            if embeddings is None:
                raise ChunkingError("Semantic chunking requires an embeddings instance")
            from langchain_experimental.text_splitter import SemanticChunker

            return SemanticChunker(embeddings)
        raise ChunkingError(f"Unsupported chunk strategy: {strategy!r}")

    async def achunk_recursive(self, documents: list[Document]) -> list[Document]:
        """Async wrapper for :meth:`chunk_recursive`."""
        return await asyncio.to_thread(self.chunk_recursive, documents)

    async def achunk_semantic(
        self,
        documents: list[Document],
        embeddings: Embeddings,
    ) -> list[Document]:
        """Async wrapper for :meth:`chunk_semantic`."""
        return await asyncio.to_thread(self.chunk_semantic, documents, embeddings)

    async def achunk(
        self,
        documents: list[Document],
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        embeddings: Embeddings | None = None,
    ) -> list[Document]:
        """Async wrapper for :meth:`chunk`."""
        return await asyncio.to_thread(self.chunk, documents, strategy, embeddings)
