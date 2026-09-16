"""Integration tests for the ChromaDB vector store.

These tests use a real ChromaDB instance backed by a temporary directory,
so they exercise the full upsert → search → delete lifecycle without
requiring a running server.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.ingestion.pipeline import ChunkMetadata, ProcessedChunk
from src.vectorstore.chroma_store import ChromaStore


def _make_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    *,
    source: str = "test.pdf",
    source_type: str = "pdf",
) -> ProcessedChunk:
    """Create a ``ProcessedChunk`` with the given content."""
    return ProcessedChunk(
        id=chunk_id,
        text=text,
        embedding=embedding,
        metadata=ChunkMetadata(
            source=source,
            source_type=source_type,
            page=1,
            section="Intro",
            ingested_at="2025-01-01T00:00:00+00:00",
            content_hash="abc123",
        ),
    )


def _fake_embedding(dim: int = 8) -> list[float]:
    """Return a unit-ish vector for testing."""
    import random

    random.seed(42)
    vec = [random.random() for _ in range(dim)]
    norm = sum(v**2 for v in vec) ** 0.5
    return [v / norm for v in vec]


@pytest.fixture()
def store(tmp_path: Path) -> ChromaStore:
    """Provide a fresh ChromaStore backed by a temp directory."""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    return ChromaStore(collection_name="test_collection", client=client)


# ── Tests ────────────────────────────────────────────────────────────────


def test_upsert_returns_count(store: ChromaStore) -> None:
    """Upsert returns the number of chunks stored."""
    chunks = [
        _make_chunk("c1", "First chunk", _fake_embedding()),
        _make_chunk("c2", "Second chunk", _fake_embedding()),
    ]
    count = asyncio.run(store.upsert(chunks))
    assert count == 2


def test_search_returns_ranked_results(store: ChromaStore) -> None:
    """Search returns results ordered by similarity."""
    vec_a = _fake_embedding()
    vec_b = _fake_embedding()
    vec_b[0] = -vec_b[0]  # flip first dimension to make it different

    chunks = [
        _make_chunk("c1", "Apple pie recipe", vec_a),
        _make_chunk("c2", "Quantum physics explained", vec_b),
    ]
    asyncio.run(store.upsert(chunks))

    results = asyncio.run(store.search(vec_a, top_k=2))
    assert len(results) == 2
    assert results[0].id == "c1"
    assert results[0].score >= results[1].score


def test_delete_removes_chunks(store: ChromaStore) -> None:
    """Deleting a chunk id removes it from the store."""
    chunks = [
        _make_chunk("c1", "Keep me", _fake_embedding()),
        _make_chunk("c2", "Delete me", _fake_embedding()),
    ]
    asyncio.run(store.upsert(chunks))
    deleted = asyncio.run(store.delete(["c2"]))
    assert deleted == 1

    remaining = asyncio.run(store.search(_fake_embedding(), top_k=10))
    assert len(remaining) == 1
    assert remaining[0].id == "c1"


def test_upsert_empty_list(store: ChromaStore) -> None:
    """Upserting an empty list is a no-op that returns 0."""
    count = asyncio.run(store.upsert([]))
    assert count == 0
