"""Unit tests for :mod:`src.ingestion.embedder`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.embedder import Embedder


class TestEmbedder:
    @pytest.mark.asyncio
    @patch("src.ingestion.embedder.OpenAIEmbeddings")
    async def test_embed_query_returns_list(self, mock_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_cls.return_value = mock_instance

        embedder = Embedder(model="test-model", embeddings=mock_instance)
        result = await embedder.embed_query("test query")

        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    @patch("src.ingestion.embedder.OpenAIEmbeddings")
    async def test_embed_documents_returns_list(self, mock_cls: MagicMock) -> None:
        from langchain_core.documents import Document

        mock_instance = MagicMock()
        mock_instance.aembed_documents = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        )
        mock_cls.return_value = mock_instance

        embedder = Embedder(model="test-model", embeddings=mock_instance)
        docs = [Document(page_content="hello"), Document(page_content="world")]
        result = await embedder.embed_documents(docs)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(vec, list) for vec in result)
