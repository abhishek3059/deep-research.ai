"""Cross-encoder reranker for post-retrieval precision."""

from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from src.config.constants import DEFAULT_TOP_K
from src.vectorstore.base import SearchResult

logger = structlog.get_logger(__name__)


class Reranker:
    """Re-rank a set of candidates using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Load the cross-encoder model.

        Args:
            model_name: Hugging Face model identifier.
        """
        self._model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        """Score each (query, document) pair and return the top_k.

        Args:
            query: Original user query.
            results: Candidate results from fusion.
            top_k: Maximum results to keep.

        Returns:
            Re-ranked list of SearchResult, highest score first.
        """
        if not results:
            return []

        pairs = [[query, r.text] for r in results]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(results, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                score=float(s),
            )
            for r, s in scored
        ]
