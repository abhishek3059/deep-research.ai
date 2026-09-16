"""Vector store module: persistent vector storage and retrieval."""


class VectorStoreError(Exception):
    """Base exception for all vector store failures."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when the vector store backend is unreachable."""


class UpsertError(VectorStoreError):
    """Raised when upserting chunks into the store fails."""


class SearchError(VectorStoreError):
    """Raised when a search query fails."""


__all__ = [
    "SearchError",
    "UpsertError",
    "VectorStoreConnectionError",
    "VectorStoreError",
]
