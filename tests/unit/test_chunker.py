"""Unit tests for :mod:`src.ingestion.chunker`."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config.constants import ChunkStrategy
from src.ingestion import ChunkingError
from src.ingestion.chunker import TextChunker


class _FakeEmbeddings(Embeddings):
    """Deterministic embeddings so semantic chunking needs no API key."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 7), float(text.count("a"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), float(text.count("a"))]


def _long_text() -> str:
    return "The quick brown fox jumps over the lazy dog. " * 20


def _semantic_text() -> str:
    return (
        "Machine learning models require large amounts of training data. "
        "Neural networks learn patterns by adjusting their weights. "
        "Bananas are yellow and grow in tropical climates. "
        "The weather forecast predicts rain for the whole weekend. "
        "Quantum computers exploit superposition and entanglement. "
        "Classical computers represent information using binary bits. "
    )


def test_chunk_recursive_with_empty_input_returns_empty_list() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=0)

    assert chunker.chunk_recursive([]) == []


def test_chunk_recursive_with_short_text_returns_single_chunk() -> None:
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    document = Document(page_content="Short content.", metadata={"source": "a.txt"})

    chunks = chunker.chunk_recursive([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short content."
    assert chunks[0].metadata["source"] == "a.txt"


def test_chunk_recursive_with_long_text_splits_into_multiple_chunks() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=0)
    document = Document(page_content=_long_text(), metadata={"source": "long.txt"})

    chunks = chunker.chunk_recursive([document])

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] == "long.txt" for chunk in chunks)


def test_chunker_with_zero_size_raises_error() -> None:
    with pytest.raises(ChunkingError, match="chunk_size must be a positive integer"):
        TextChunker(chunk_size=0, chunk_overlap=0)


def test_chunker_with_negative_overlap_raises_error() -> None:
    with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)


def test_chunk_with_overlap_not_smaller_than_size_raises_chunking_error() -> None:
    with pytest.raises(ChunkingError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_semantic_with_fake_embeddings_produces_chunks() -> None:
    chunker = TextChunker()
    document = Document(page_content=_semantic_text(), metadata={"source": "s.md"})

    chunks = chunker.chunk_semantic([document], embeddings=_FakeEmbeddings())

    assert chunks
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.metadata["source"] == "s.md" for chunk in chunks)
