"""Ingestion pipeline orchestrator.

Coordinates loading, chunking, deduplication, and embedding, and produces
:class:`ProcessedChunk` objects matching the ingestion -> vector store
contract (docs/CONTRACTS.md section 4.1).

Ordering note: deduplication runs *after* chunking but *before* embedding.
This mirrors the contract's dataclasses, removes duplicate content, and
avoids paying to embed the same chunk twice. The high-level sequence is
therefore load -> chunk -> dedup -> embed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.config.constants import SourceType
from src.ingestion import IngestionError
from src.ingestion.chunker import TextChunker
from src.ingestion.deduplicator import Deduplicator
from src.ingestion.embedder import Embedder
from src.ingestion.loaders import DocumentLoader, detect_source_type

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata attached to every processed chunk.

    Matches docs/CONTRACTS.md section 4.1.
    """

    source: str
    """File path or URL the chunk originated from."""

    source_type: str
    """One of ``pdf`` | ``web`` | ``markdown`` | ``csv`` | ``text``."""

    page: int | None
    """Page number for PDFs, otherwise ``None``."""

    section: str | None
    """Section header if one was detected, otherwise ``None``."""

    ingested_at: str
    """ISO 8601 timestamp of ingestion."""

    content_hash: str
    """SHA-256 of the raw chunk text."""


@dataclass
class ProcessedChunk:
    """A chunk ready to be stored in the vector store.

    Matches docs/CONTRACTS.md section 4.1.
    """

    id: str
    """Deterministic hash of content + source."""

    text: str
    """Chunk text content."""

    embedding: list[float]
    """Dense embedding vector."""

    metadata: ChunkMetadata
    """Source, page, timestamp, etc."""


class IngestionPipeline:
    """Orchestrate load -> chunk -> dedup -> embed for documents."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embedder: Embedder | None = None,
        deduplicator: Deduplicator | None = None,
    ) -> None:
        """Wire the pipeline stages, defaulting each to a real implementation.

        Args:
            loader: Document loader stage.
            chunker: Chunking stage.
            embedder: Embedding stage.
            deduplicator: Deduplication stage.
        """
        self._loader = loader or DocumentLoader()
        self._chunker = chunker or TextChunker()
        self._embedder = embedder or Embedder()
        self._deduplicator = deduplicator or Deduplicator()

    async def ingest(
        self,
        source: str,
        source_type: SourceType | None = None,
    ) -> list[ProcessedChunk]:
        """Run the full pipeline for a single source.

        Args:
            source: Local file path or HTTP(S) URL to ingest.
            source_type: Override auto-detected source type.

        Returns:
            Processed chunks with embeddings and contract metadata.

        Raises:
            IngestionError: If a stage fails or counts are inconsistent.
        """
        started = time.perf_counter()
        documents = await self._loader.load(source, source_type)
        chunks = await self._chunker.achunk_recursive(documents)
        unique_chunks = self._deduplicator.deduplicate(chunks)
        embeddings = await self._embedder.embed_documents(unique_chunks)
        processed = self._build_processed_chunks(unique_chunks, embeddings, source)
        logger.info(
            "Ingestion completed",
            source=source,
            document_count=len(documents),
            chunk_count=len(processed),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return processed

    async def ingest_many(self, sources: list[str]) -> list[ProcessedChunk]:
        """Ingest several sources sequentially.

        Args:
            sources: File paths and/or URLs.

        Returns:
            Concatenated processed chunks across all sources.
        """
        processed: list[ProcessedChunk] = []
        for source in sources:
            processed.extend(await self.ingest(source))
        return processed

    def _build_processed_chunks(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        source: str,
    ) -> list[ProcessedChunk]:
        """Pair chunks with their embeddings and attach contract metadata."""
        if len(documents) != len(embeddings):
            raise IngestionError(
                "Document/embedding count mismatch: "
                f"{len(documents)} != {len(embeddings)}"
            )
        source_type = str(detect_source_type(source))
        ingested_at = datetime.now(UTC).isoformat()
        return [
            _to_processed_chunk(document, vector, source, source_type, ingested_at)
            for document, vector in zip(documents, embeddings, strict=True)
        ]


def _to_processed_chunk(
    document: Document,
    embedding: list[float],
    source: str,
    source_type: str,
    ingested_at: str,
) -> ProcessedChunk:
    """Convert a LangChain document plus its embedding into a chunk."""
    text = document.page_content
    content_hash = _sha256(text)
    raw_metadata = document.metadata or {}
    chunk_source = str(raw_metadata.get("source", source))
    metadata = ChunkMetadata(
        source=chunk_source,
        source_type=source_type,
        page=_optional_int(raw_metadata.get("page")),
        section=_optional_str(raw_metadata.get("section")),
        ingested_at=ingested_at,
        content_hash=content_hash,
    )
    return ProcessedChunk(
        id=_chunk_id(content_hash, chunk_source),
        text=text,
        embedding=embedding,
        metadata=metadata,
    )


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(content_hash: str, source: str) -> str:
    """Build a deterministic chunk id from content hash and source."""
    return hashlib.sha256(f"{source}:{content_hash}".encode()).hexdigest()


def _optional_int(value: object) -> int | None:
    """Coerce a metadata value to ``int`` when it is a genuine integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_str(value: object) -> str | None:
    """Coerce a non-empty metadata value to ``str``, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None
