"""Constants and enums used across the project."""

from enum import StrEnum


class SourceType(StrEnum):
    """Document source types supported by the ingestion pipeline."""

    PDF = "pdf"
    WEB = "web"
    MARKDOWN = "markdown"
    CSV = "csv"
    TEXT = "text"


class VectorStoreType(StrEnum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    PINECONE = "pinecone"


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ChunkStrategy(StrEnum):
    """Chunking strategies."""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# Defaults
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5
MAX_ITERATIONS = 3
COLLECTION_NAME = "research_documents"
