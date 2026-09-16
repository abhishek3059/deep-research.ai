"""Unit tests for :mod:`src.ingestion.loaders`."""

from __future__ import annotations

import pytest

from src.config.constants import SourceType
from src.ingestion.loaders import DocumentLoader, detect_source_type


class TestDetectSourceType:
    def test_auto_detect_txt_file(self) -> None:
        assert detect_source_type("document.txt") is SourceType.TEXT

    def test_auto_detect_pdf_file(self) -> None:
        assert detect_source_type("report.pdf") is SourceType.PDF

    def test_auto_detect_csv_file(self) -> None:
        assert detect_source_type("data.csv") is SourceType.CSV

    def test_auto_detect_markdown_file(self) -> None:
        assert detect_source_type("readme.md") is SourceType.MARKDOWN

    def test_auto_detect_http_url(self) -> None:
        assert detect_source_type("https://example.com") is SourceType.WEB

    def test_auto_detect_unsupported_extension_raises(self) -> None:
        from src.ingestion import UnsupportedSourceError

        with pytest.raises(UnsupportedSourceError):
            detect_source_type("image.png")


class TestDocumentLoader:
    @pytest.mark.asyncio
    async def test_load_text_file(self) -> None:
        loader = DocumentLoader()
        documents = await loader.load("tests/fixtures/sample.txt")

        assert len(documents) > 0
        assert "sample document" in documents[0].page_content.lower()
