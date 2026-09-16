"""Ingestion module: load, chunk, embed, and deduplicate documents."""


class IngestionError(Exception):
    """Base exception for all ingestion failures."""


class UnsupportedSourceError(IngestionError):
    """Raised when a source path or URL has no supported loader."""


class ChunkingError(IngestionError):
    """Raised when document chunking cannot be performed."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


__all__ = [
    "ChunkingError",
    "EmbeddingError",
    "IngestionError",
    "UnsupportedSourceError",
]
