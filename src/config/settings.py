"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Application
    log_level: str = "INFO"


settings = Settings()
