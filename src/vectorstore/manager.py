"""Vector store manager — factory for obtaining the correct store instance.

Respects the ``VECTOR_STORE_TYPE`` environment variable and provides a
singleton accessor so the rest of the app never instantiates a store directly.
"""

from __future__ import annotations

import structlog

from src.config.constants import VectorStoreType
from src.config.settings import settings
from src.vectorstore.base import VectorStoreProtocol
from src.vectorstore.chroma_store import ChromaStore

logger = structlog.get_logger(__name__)

_store: VectorStoreProtocol | None = None


def get_store() -> VectorStoreProtocol:
    """Return the configured vector store singleton.

    Uses ``settings.vector_store_type`` to decide which backend to
    instantiate.  Only ChromaDB is fully implemented; Pinecone is a stub
    that raises ``NotImplementedError``.
    """
    global _store
    if _store is not None:
        return _store

    store_type = settings.vector_store_type
    logger.info("Creating vector store", backend=store_type)

    if store_type == VectorStoreType.CHROMA:
        _store = ChromaStore()
    elif store_type == VectorStoreType.PINECONE:
        from src.vectorstore.pinecone_store import PineconeStore

        _store = PineconeStore()
    else:
        raise ValueError(f"Unsupported vector store type: {store_type!r}")

    return _store


def reset_store() -> None:
    """Reset the singleton (useful in tests)."""
    global _store
    _store = None
