"""Unit tests for the post-retrieval coverage floor.

The floor judges relevance against the real corpus (top-1 dense cosine
similarity) instead of pre-retrieval keyword matching. Below the floor the
pipeline returns no results with truthful counts so callers answer honestly.
"""

from __future__ import annotations

import pytest

from src.config.constants import DEFAULT_MIN_COVERAGE
from src.ingestion.pipeline import ChunkMetadata
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.base import SearchResult


class _StubStore:
    """Vector store returning one hit with a fixed similarity score."""

    def __init__(self, score: float) -> None:
        self._score = score

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        meta = ChunkMetadata(
            source="doc.md",
            source_type="markdown",
            page=None,
            section=None,
            ingested_at="2026-01-01T00:00:00Z",
            content_hash="abc",
        )
        return [SearchResult(id="c1", text="Some content", metadata=meta, score=self._score)]


class _EmptyStore:
    """Vector store with nothing indexed."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        return []


class _StubEmbedder:
    """Embedder returning a fixed unit vector (no API keys)."""

    async def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def _make_pipeline(store: object, **kwargs: object) -> RetrievalPipeline:
    return RetrievalPipeline(
        vector_store=store,  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
        enable_sparse=False,
        enable_rerank=False,
        enable_multi_query=False,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_retrieve_with_strong_match_returns_results() -> None:
    result = await _make_pipeline(_StubStore(score=0.9)).retrieve("What is RAG?")
    assert len(result.results) == 1
    assert result.retrieval_metadata.dense_results == 1


@pytest.mark.asyncio
async def test_retrieve_with_weak_match_returns_empty_with_truthful_counts() -> None:
    result = await _make_pipeline(_StubStore(score=0.05)).retrieve("Lottery numbers?")
    assert result.results == []
    # Counts are measured, not falsified: dense ran and found one weak hit.
    assert result.retrieval_metadata.dense_results == 1
    assert result.query == "Lottery numbers?"


@pytest.mark.asyncio
async def test_retrieve_with_floor_disabled_passes_weak_match() -> None:
    result = await _make_pipeline(_StubStore(score=0.01), min_coverage=0.0).retrieve("Anything?")
    assert len(result.results) == 1


@pytest.mark.asyncio
async def test_retrieve_with_empty_store_returns_empty_without_error() -> None:
    result = await _make_pipeline(_EmptyStore()).retrieve("Anything?")
    assert result.results == []
    assert result.retrieval_metadata.dense_results == 0


def test_default_floor_is_conservative_and_enabled() -> None:
    assert DEFAULT_MIN_COVERAGE == 0.25
    pipeline = _make_pipeline(_StubStore(score=0.9))
    assert pipeline._min_coverage == DEFAULT_MIN_COVERAGE
