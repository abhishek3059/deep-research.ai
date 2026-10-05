"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.constants import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_MIN_COVERAGE,
    GEMINI_OPENAI_BASE_URL,
)


class Settings(BaseSettings):
    """Central configuration for DeepResearch AI.

    All values are loaded from environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Embedding provider selection. Both implementations satisfy
    # EmbedderProtocol, so swapping is a settings change (ADR-012).
    embedding_provider: str = "openai"  # "openai" | "gemini"
    gemini_embedding_model: str = DEFAULT_GEMINI_EMBEDDING_MODEL
    # Google serves chat over an OpenAI-compatible endpoint; embeddings go
    # through the native SDK because that compatibility surface omits them.
    gemini_openai_base_url: str = GEMINI_OPENAI_BASE_URL

    # Vector Store
    pinecone_api_key: str = ""
    vector_store_type: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # CORS
    cors_origins: list[str] = ["http://localhost:8501"]

    # Retrieval
    top_k: int = 5
    min_coverage: float = DEFAULT_MIN_COVERAGE

    # Application
    log_level: str = "INFO"


settings = Settings()
