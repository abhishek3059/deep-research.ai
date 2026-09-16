"""Unit tests for :mod:`src.ingestion.deduplicator`."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document

from src.ingestion.deduplicator import Deduplicator


def test_deduplicate_with_empty_input_returns_empty_list() -> None:
    deduplicator = Deduplicator()

    assert deduplicator.deduplicate([]) == []


def test_deduplicate_with_unique_documents_returns_all_documents() -> None:
    deduplicator = Deduplicator()
    documents = [
        Document(page_content="alpha", metadata={"source": "a.txt"}),
        Document(page_content="beta", metadata={"source": "b.txt"}),
    ]

    result = deduplicator.deduplicate(documents)

    assert result == documents


def test_deduplicate_with_duplicate_content_keeps_first_occurrence() -> None:
    deduplicator = Deduplicator()
    documents = [
        Document(page_content="same", metadata={"source": "first.txt"}),
        Document(page_content="different", metadata={"source": "other.txt"}),
        Document(page_content="same", metadata={"source": "second.txt"}),
    ]

    result = deduplicator.deduplicate(documents)

    assert [document.page_content for document in result] == ["same", "different"]
    assert result[0].metadata["source"] == "first.txt"


def test_compute_hash_with_known_content_returns_sha256_hexdigest() -> None:
    expected = hashlib.sha256(b"hello").hexdigest()

    assert Deduplicator.compute_hash("hello") == expected
    assert Deduplicator.compute_hash("hello") != Deduplicator.compute_hash("world")
