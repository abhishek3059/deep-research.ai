"""Tests for RRF fusion in the HybridRetriever."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from src.ingestion.pipeline import ChunkMetadata
from src.retrieval.hybrid import HybridRetriever
from src.vectorstore.base import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(source: str = "test.pdf") -> ChunkMetadata:
    return ChunkMetadata(
        source=source,
        source_type="pdf",
        page=None,
        section=None,
        ingested_at="2025-01-01T00:00:00+00:00",
        content_hash="abc123",
    )


def _result(doc_id: str, text: str = "placeholder") -> SearchResult:
    return SearchResult(id=doc_id, text=text, metadata=_make_metadata(), score=0.0)


class _FakeDense:
    """Return a fixed ordered list of results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        return self._results[:top_k]


class _FakeSparse:
    """Return a fixed ordered list of results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rrf_score_formula() -> None:
    """Verify the basic RRF formula: 1 / (k + rank)."""
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._k = 60
    # rank 1 → 1/61
    assert math.isclose(retriever._rrf_score(1), 1 / 61, rel_tol=1e-9)
    # rank 2 → 1/62
    assert math.isclose(retriever._rrf_score(2), 1 / 62, rel_tol=1e-9)
    # rank 10 → 1/70
    assert math.isclose(retriever._rrf_score(10), 1 / 70, rel_tol=1e-9)


def test_rrf_fusion_merges_results() -> None:
    """Dense and sparse results that partially overlap are merged correctly."""
    dense = [_result("d1"), _result("d2"), _result("d3")]
    sparse = [_result("d2"), _result("d3"), _result("d4")]

    dense_f = _FakeDense(dense)
    sparse_f = _FakeSparse(sparse)

    async def _run() -> None:
        hybrid = HybridRetriever(dense_f, sparse_f, k=60)
        results = await hybrid.retrieve("query", [0.1], top_k=5)
        ids = [r.id for r in results]
        assert "d1" in ids
        assert "d4" in ids
        # d2 and d3 appear in both lists → should have the highest scores
        assert ids[0] in {"d2", "d3"}

    import asyncio

    asyncio.run(_run())


def test_rrf_fusion_with_disjoint_results() -> None:
    """When dense and sparse return completely different documents, all are included."""
    dense = [_result("d1"), _result("d2")]
    sparse = [_result("s1"), _result("s2")]

    dense_f = _FakeDense(dense)
    sparse_f = _FakeSparse(sparse)

    async def _run() -> None:
        hybrid = HybridRetriever(dense_f, sparse_f, k=60)
        results = await hybrid.retrieve("query", [0.1], top_k=10)
        ids = [r.id for r in results]
        assert set(ids) == {"d1", "d2", "s1", "s2"}

    import asyncio

    asyncio.run(_run())


def test_rrf_with_equal_ranks() -> None:
    """When a document appears at the same rank in both lists, its score is doubled."""
    dense = [_result("a"), _result("b")]
    sparse = [_result("a"), _result("b")]

    dense_f = _FakeDense(dense)
    sparse_f = _FakeSparse(sparse)

    async def _run() -> None:
        hybrid = HybridRetriever(dense_f, sparse_f, k=60)
        results = await hybrid.retrieve("query", [0.1], top_k=2)
        # Both appear at rank 1 in both lists → score = 2 * 1/61
        assert math.isclose(results[0].score, 2 / 61, rel_tol=1e-9)
        # Both appear at rank 2 in both lists → score = 2 * 1/62
        assert math.isclose(results[1].score, 2 / 62, rel_tol=1e-9)

    import asyncio

    asyncio.run(_run())
