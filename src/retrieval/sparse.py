"""Sparse BM25 retriever for keyword-based search."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.config.constants import DEFAULT_TOP_K
from src.ingestion.pipeline import ChunkMetadata
from src.vectorstore.base import SearchResult


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return [t for t in re.split(r"\W+", text.lower()) if t]


class SparseRetriever:
    """BM25Okapi-based sparse retriever.

    Operates on in-memory documents rather than a vector store.
    """

    def __init__(self, documents: list[str], metadatas: list[ChunkMetadata]) -> None:
        """Initialise the BM25 index.

        Args:
            documents: Raw text strings (one per chunk).
            metadatas: Corresponding ChunkMetadata for each document.
        """
        if len(documents) != len(metadatas):
            raise ValueError(
                f"Document/metadata count mismatch: {len(documents)} != {len(metadatas)}"
            )
        self._documents = documents
        self._metadatas = metadatas
        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Rank documents by BM25 score and return the top_k.

        Args:
            query: Natural language query.
            top_k: Maximum results to return.

        Returns:
            Ranked list of SearchResult, highest relevance first.
        """
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                id=f"sparse-{idx}",
                text=self._documents[idx],
                metadata=self._metadatas[idx],
                score=float(scores[idx]),
            )
            for idx in ranked_indices
        ]
