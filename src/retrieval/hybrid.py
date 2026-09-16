"""Hybrid retriever combining dense and sparse results via Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from src.config.constants import DEFAULT_TOP_K
from src.vectorstore.base import SearchResult

if TYPE_CHECKING:
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever


class HybridRetriever:
    """Merges dense and sparse retrieval using Reciprocal Rank Fusion (RRF).

    score(d) = sum( 1 / (k + rank_i(d)) ) for each ranking source i.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        k: int = 60,
    ) -> None:
        """Wire the two sub-retrievers and set the RRF constant.

        Args:
            dense_retriever: Embedding-based retriever.
            sparse_retriever: BM25-based retriever.
            k: RRF smoothing constant (default 60, per the original paper).
        """
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._k = k

    def _rrf_score(self, rank: int) -> float:
        """Compute the RRF contribution for a single rank (1-indexed)."""
        return 1.0 / (self._k + rank)

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Run dense + sparse retrieval and fuse via RRF.

        Args:
            query: Natural language query.
            query_embedding: Dense embedding of *query*.
            top_k: Maximum results to return.

        Returns:
            Fused, ranked list of SearchResult.
        """
        dense_results = await self._dense.retrieve(query_embedding, top_k=top_k * 2)
        sparse_results = await self._sparse.retrieve(query, top_k=top_k * 2)

        scores: dict[str, float] = defaultdict(float)
        result_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            scores[result.id] += self._rrf_score(rank)
            result_map[result.id] = result

        for rank, result in enumerate(sparse_results, start=1):
            scores[result.id] += self._rrf_score(rank)
            result_map[result.id] = result

        ranked_ids = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:top_k]

        return [
            SearchResult(
                id=rid,
                text=result_map[rid].text,
                metadata=result_map[rid].metadata,
                score=scores[rid],
            )
            for rid in ranked_ids
        ]
