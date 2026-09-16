"""ChromaDB vector store implementation.

Uses ``asyncio.to_thread`` to wrap ChromaDB's synchronous API so that the
rest of the application can ``await`` all store operations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import chromadb
import structlog

from src.config.constants import COLLECTION_NAME
from src.config.settings import settings
from src.ingestion.pipeline import ChunkMetadata, ProcessedChunk
from src.vectorstore import SearchError, UpsertError, VectorStoreConnectionError, VectorStoreError
from src.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)

# Module-level client — created once and reused across the process.
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client, creating it on first call."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


class ChromaStore:
    """ChromaDB-backed vector store conforming to VectorStoreProtocol."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        """Initialise the ChromaDB collection.

        Args:
            collection_name: Name of the collection to use.
            client: Optional pre-configured ChromaDB client (useful in tests).
        """
        self._client = client or _get_client()
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to ChromaDB collection '{collection_name}'"
            ) from exc
        logger.info("ChromaDB collection ready", collection=collection_name)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the collection by vector similarity.

        Args:
            query_embedding: Dense embedding of the query.
            top_k: Maximum results to return.
            filters: Optional ``where`` filter dict passed to ChromaDB.

        Returns:
            Ranked list of ``SearchResult``, highest similarity first.
        """
        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
            )
        except Exception as exc:
            raise SearchError("ChromaDB query failed") from exc

        ids: list[str] = results.get("ids", [[]])[0]
        documents: list[str] = results.get("documents", [[]])[0]
        distances: list[float] = results.get("distances", [[]])[0]
        metadatas: list[dict[str, Any]] = results.get("metadatas", [[]])[0]

        search_results: list[SearchResult] = []
        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas, strict=True):
            search_results.append(
                SearchResult(
                    id=doc_id,
                    text=doc,
                    metadata=_meta_from_dict(meta),
                    # ChromaDB cosine distance is in [0, 2] — convert to similarity
                    score=max(0.0, 1.0 - dist / 2.0),
                )
            )
        return search_results

    async def upsert(self, chunks: list[ProcessedChunk]) -> int:
        """Insert or update chunks in the collection.

        Args:
            chunks: Processed chunks with embeddings and metadata.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = [_meta_to_dict(c.metadata) for c in chunks]

        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise UpsertError("ChromaDB upsert failed") from exc

        logger.info("Upserted chunks", count=len(ids))
        return len(ids)

    async def delete(self, ids: list[str]) -> int:
        """Delete chunks by id.

        Args:
            ids: Chunk ids to remove.

        Returns:
            Number of chunks deleted.
        """
        if not ids:
            return 0

        try:
            await asyncio.to_thread(self._collection.delete, ids=ids)
        except Exception as exc:
            raise VectorStoreError("ChromaDB delete failed") from exc

        logger.info("Deleted chunks", count=len(ids))
        return len(ids)

    async def count(self) -> int:
        """Return the number of items in the collection."""
        return await asyncio.to_thread(self._collection.count)


def _meta_to_dict(meta: ChunkMetadata) -> dict[str, Any]:
    """Serialise a ``ChunkMetadata`` to a flat dict for ChromaDB storage."""
    return {
        "source": meta.source,
        "source_type": meta.source_type,
        "page": meta.page if meta.page is not None else -1,
        "section": meta.section or "",
        "ingested_at": meta.ingested_at,
        "content_hash": meta.content_hash,
    }


def _meta_from_dict(d: dict[str, Any]) -> ChunkMetadata:
    """Deserialise a flat dict back into a ``ChunkMetadata``."""
    page = d.get("page", -1)
    section = d.get("section", "") or None
    return ChunkMetadata(
        source=d["source"],
        source_type=d["source_type"],
        page=page if page != -1 else None,
        section=section,
        ingested_at=d["ingested_at"],
        content_hash=d["content_hash"],
    )
