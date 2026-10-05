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

# Minimum top-1 dense cosine similarity for retrieved results to count as
# coverage. Below this, the corpus is judged not to cover the query and the
# pipeline answers honestly instead of generating. Conservative by design:
# unrelated content scores near 0.0–0.2, related content well above 0.4.
DEFAULT_MIN_COVERAGE = 0.25

# ---------------------------------------------------------------------------
# Provider endpoints
# ---------------------------------------------------------------------------
# Google exposes an OpenAI-compatible chat endpoint, so the LLM provider reaches
# Gemini through ChatOpenAI with a different base_url. That keeps the chat side a
# configuration change rather than a new code path (see ADR-012).
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Default Gemini embedding model. Google's OpenAI-compatible surface does NOT
# cover embeddings, so these go through the native SDK instead.
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
