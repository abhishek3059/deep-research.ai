"""Unit tests for :mod:`src.ingestion.pipeline` — dataclass contracts."""

from __future__ import annotations

from src.ingestion.pipeline import ChunkMetadata, ProcessedChunk


class TestChunkMetadata:
    def test_chunk_metadata_dataclass_creation(self) -> None:
        metadata = ChunkMetadata(
            source="doc.pdf",
            source_type="pdf",
            page=1,
            section="Introduction",
            ingested_at="2026-01-01T00:00:00+00:00",
            content_hash="abc123",
        )

        assert metadata.source == "doc.pdf"
        assert metadata.source_type == "pdf"
        assert metadata.page == 1
        assert metadata.section == "Introduction"
        assert metadata.content_hash == "abc123"

    def test_chunk_metadata_optional_fields(self) -> None:
        metadata = ChunkMetadata(
            source="data.csv",
            source_type="csv",
            page=None,
            section=None,
            ingested_at="2026-01-01T00:00:00+00:00",
            content_hash="def456",
        )

        assert metadata.page is None
        assert metadata.section is None


class TestProcessedChunk:
    def test_processed_chunk_dataclass_creation(self) -> None:
        metadata = ChunkMetadata(
            source="doc.txt",
            source_type="text",
            page=None,
            section=None,
            ingested_at="2026-01-01T00:00:00+00:00",
            content_hash="aaa",
        )
        chunk = ProcessedChunk(
            id="chunk-id-1",
            text="Hello world",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata,
        )

        assert chunk.id == "chunk-id-1"
        assert chunk.text == "Hello world"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata.source == "doc.txt"
