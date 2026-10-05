"""Retrieval pipeline orchestrator.

Coordinates multi-query expansion, dense+sparse retrieval, RRF fusion,
cross-encoder reranking, and returns a contract-compliant :class:`RetrievalResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from src.config.constants import DEFAULT_MIN_COVERAGE, DEFAULT_TOP_K
from src.ingestion.embedder_base import EmbedderProtocol, build_embedder
from src.ingestion.pipeline import ChunkMetadata
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.multi_query import MultiQueryExpander, QueryLLM
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.vectorstore.base import SearchResult

if TYPE_CHECKING:
    from src.vectorstore.base import VectorStoreProtocol

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalMeta:
    """Metadata about how retrieval was performed.

    Matches docs/CONTRACTS.md section 4.3.
    """

    strategy: str
    dense_results: int
    sparse_results: int
    reranked: bool
    latency_ms: float


@dataclass
class RetrievalResult:
    """Final output of the retrieval pipeline.

    Matches docs/CONTRACTS.md section 4.3.
    """

    query: str
    expanded_queries: list[str]
    results: list[SearchResult]
    retrieval_metadata: RetrievalMeta


class RetrievalPipeline:
    """End-to-end retrieval orchestrator.

    Workflow: expand -> dense + sparse -> RRF -> rerank -> RetrievalResult
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        embedder: EmbedderProtocol | None = None,
        documents: list[str] | None = None,
        metadatas: list[ChunkMetadata] | None = None,
        enable_sparse: bool = True,
        enable_rerank: bool = True,
        enable_multi_query: bool = True,
        llm: QueryLLM | None = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> None:
        """Wire pipeline stages.

        Args:
            vector_store: Backend implementing VectorStoreProtocol.
            embedder: EmbedderProtocol used to encode the query before dense retrieval.
            documents: Raw texts for BM25 index (required when *enable_sparse*).
            metadatas: ChunkMetadata list for sparse results.
            enable_sparse: Whether to run BM25 alongside dense retrieval.
            enable_rerank: Whether to apply cross-encoder reranking.
            enable_multi_query: Whether to expand the query.
            llm: Optional LLM for multi-query expansion.
            min_coverage: Minimum top-1 dense cosine similarity for the
                corpus to count as covering the query. Below this the
                pipeline returns no results (with truthful counts) so
                callers answer honestly instead of generating. ``0.0``
                disables the floor. Relevance is judged here — against the
                real corpus — never by pre-retrieval keyword matching.
        """
        self._vector_store = vector_store
        self._embedder = embedder or build_embedder()
        self._enable_sparse = enable_sparse
        self._enable_rerank = enable_rerank
        self._enable_multi_query = enable_multi_query
        self._min_coverage = min_coverage

        self._dense = DenseRetriever(vector_store)
        self._sparse: SparseRetriever | None = None
        if enable_sparse and documents is not None and metadatas is not None:
            self._sparse = SparseRetriever(documents, metadatas)

        self._reranker: Reranker | None = None
        if enable_rerank:
            self._reranker = Reranker()

        self._expander = MultiQueryExpander(llm=llm)

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        """Run the full retrieval pipeline.

        Args:
            query: User's natural language question.
            top_k: Maximum results to return.

        Returns:
            Contract-compliant RetrievalResult.
        """
        started = time.perf_counter()

        # 1. Multi-query expansion
        if self._enable_multi_query:
            queries = self._expander.expand(query)
        else:
            queries = [query]

        # 2. Dense retrieval — embed the query first
        query_embedding = await self._embedder.embed_query(query)
        dense_results = await self._dense.retrieve(
            query_embedding=query_embedding,
            top_k=top_k * 2 if self._sparse else top_k,
        )
        dense_count = len(dense_results)

        # Coverage floor: the top-1 dense cosine similarity measures whether
        # the corpus covers the query at all. Below the floor, return no
        # results (counts stay truthful) so callers answer honestly instead
        # of generating from noise. Set min_coverage=0.0 to disable.
        top_dense = dense_results[0].score if dense_results else 0.0
        if self._min_coverage > 0 and top_dense < self._min_coverage:
            logger.info(
                "Coverage floor tripped — corpus does not cover query",
                top_dense_score=round(top_dense, 4),
                floor=self._min_coverage,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return RetrievalResult(
                query=query,
                expanded_queries=queries,
                results=[],
                retrieval_metadata=RetrievalMeta(
                    strategy="hybrid" if self._sparse is not None else "dense",
                    dense_results=dense_count,
                    sparse_results=0,
                    reranked=False,
                    latency_ms=elapsed_ms,
                ),
            )

        # 3. Sparse retrieval
        sparse_results: list[SearchResult] = []
        if self._sparse is not None:
            sparse_results = await self._sparse.retrieve(query, top_k=top_k * 2)
        sparse_count = len(sparse_results)

        # 4. RRF fusion
        if self._sparse is not None and sparse_results:
            hybrid = HybridRetriever(self._dense, self._sparse)
            fused = await hybrid.retrieve(query, query_embedding=query_embedding, top_k=top_k)
        else:
            fused = dense_results[:top_k]

        # 5. Rerank
        reranked = False
        if self._reranker is not None and fused:
            fused = await self._reranker.rerank(query, fused, top_k=top_k)
            reranked = True

        elapsed_ms = (time.perf_counter() - started) * 1000

        strategy = "hybrid" if self._sparse is not None else "dense"
        meta = RetrievalMeta(
            strategy=strategy,
            dense_results=dense_count,
            sparse_results=sparse_count,
            reranked=reranked,
            latency_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Retrieval completed",
            strategy=strategy,
            total_results=len(fused),
            latency_ms=meta.latency_ms,
        )

        return RetrievalResult(
            query=query,
            expanded_queries=queries,
            results=fused,
            retrieval_metadata=meta,
        )
