"""Unit tests for :mod:`src.ingestion.gemini_embedder` and the provider factory.

Covers ADR-012: embeddings are swappable behind a structural protocol, and
provider selection happens in exactly one place.

No network access and no API key required — every client is stubbed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import settings
from src.ingestion import EmbeddingError
from src.ingestion.embedder_base import EmbedderProtocol, build_embedder
from src.ingestion.gemini_embedder import GeminiEmbedder


def _client() -> MagicMock:
    """A stubbed GoogleGenerativeAIEmbeddings with async methods."""
    instance = MagicMock()
    instance.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    instance.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    instance.model = "models/text-embedding-004"
    return instance


class TestGeminiEmbedder:
    @pytest.mark.asyncio
    async def test_embed_query_returns_client_vector(self) -> None:
        client = _client()
        embedder = GeminiEmbedder(embeddings=client)

        result = await embedder.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        client.aembed_query.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_embed_texts_returns_one_vector_per_input_in_order(self) -> None:
        client = _client()
        embedder = GeminiEmbedder(embeddings=client)

        result = await embedder.embed_texts(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_texts_with_empty_input_short_circuits(self) -> None:
        client = _client()
        embedder = GeminiEmbedder(embeddings=client)

        assert await embedder.embed_texts([]) == []
        client.aembed_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embed_texts_batches_by_batch_size(self) -> None:
        client = _client()
        client.aembed_documents = AsyncMock(
            side_effect=[[[0.1]], [[0.2]], [[0.3]]],
        )
        embedder = GeminiEmbedder(embeddings=client, batch_size=1)

        result = await embedder.embed_texts(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]
        assert client.aembed_documents.await_count == 3

    @pytest.mark.asyncio
    async def test_embed_documents_uses_page_content(self) -> None:
        from langchain_core.documents import Document

        client = _client()
        embedder = GeminiEmbedder(embeddings=client)

        await embedder.embed_documents([Document(page_content="hello world")])

        assert client.aembed_documents.await_args[0][0] == ["hello world"]

    @pytest.mark.asyncio
    async def test_query_failure_is_translated_to_embedding_error(self) -> None:
        client = _client()
        client.aembed_query = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        embedder = GeminiEmbedder(embeddings=client)

        with pytest.raises(EmbeddingError, match="Query embedding failed"):
            await embedder.embed_query("x")

    @pytest.mark.asyncio
    async def test_batch_failure_is_translated_to_embedding_error(self) -> None:
        client = _client()
        client.aembed_documents = AsyncMock(side_effect=RuntimeError("bad request"))
        embedder = GeminiEmbedder(embeddings=client)

        with pytest.raises(EmbeddingError, match="Embedding request failed"):
            await embedder.embed_texts(["a"])

    def test_model_name_reads_from_client(self) -> None:
        assert GeminiEmbedder(embeddings=_client()).model_name == "models/text-embedding-004"

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="positive integer"):
            GeminiEmbedder(embeddings=_client(), batch_size=0)


class TestGeminiEmbedderClientConstruction:
    @patch("src.ingestion.gemini_embedder.GoogleGenerativeAIEmbeddings")
    def test_builds_client_from_settings(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = _client()

        with patch.object(settings, "gemini_api_key", "test-key"):
            GeminiEmbedder()

        assert mock_cls.call_args.kwargs["google_api_key"] == "test-key"
        assert mock_cls.call_args.kwargs["model"] == settings.gemini_embedding_model

    @patch("src.ingestion.gemini_embedder.GoogleGenerativeAIEmbeddings")
    def test_missing_api_key_fails_loudly_at_construction(self, mock_cls: MagicMock) -> None:
        """A missing key must raise immediately, not on the first query."""
        with (
            patch.object(settings, "gemini_api_key", ""),
            pytest.raises(EmbeddingError, match="GEMINI_API_KEY"),
        ):
            GeminiEmbedder()
        mock_cls.assert_not_called()


class TestProviderFactory:
    """Provider selection must live in exactly one place (ADR-012)."""

    def test_openai_provider_builds_embedder(self) -> None:
        with (
            patch("src.ingestion.embedder.OpenAIEmbeddings") as mock_cls,
            patch.object(settings, "embedding_provider", "openai"),
        ):
            mock_cls.return_value = MagicMock()
            result = build_embedder()

        assert isinstance(result, EmbedderProtocol)

    def test_gemini_provider_builds_embedder(self) -> None:
        with (
            patch("src.ingestion.gemini_embedder.GoogleGenerativeAIEmbeddings") as mock_cls,
            patch.object(settings, "gemini_api_key", "test-key"),
            patch.object(settings, "embedding_provider", "gemini"),
        ):
            mock_cls.return_value = _client()
            result = build_embedder()

        assert isinstance(result, EmbedderProtocol)

    def test_explicit_argument_overrides_settings(self) -> None:
        with (
            patch("src.ingestion.gemini_embedder.GoogleGenerativeAIEmbeddings") as mock_cls,
            patch.object(settings, "gemini_api_key", "test-key"),
        ):
            mock_cls.return_value = _client()
            result = build_embedder("gemini")

        assert isinstance(result, EmbedderProtocol)

    def test_provider_name_is_case_insensitive(self) -> None:
        with (
            patch("src.ingestion.gemini_embedder.GoogleGenerativeAIEmbeddings") as mock_cls,
            patch.object(settings, "gemini_api_key", "test-key"),
        ):
            mock_cls.return_value = _client()
            assert isinstance(build_embedder("GEMINI"), EmbedderProtocol)

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            build_embedder("nope")


class TestProtocolConformance:
    """The point of ADR-012: existing implementations conform with no changes.

    Note: ``issubclass()`` is unavailable here because ``EmbedderProtocol``
    declares a non-method member (``model_name``); Python only supports
    ``issubclass`` on all-method protocols. ``isinstance()`` is the correct
    structural check, and nominal non-inheritance is asserted via the MRO.
    """

    def test_openai_embedder_satisfies_protocol_without_inheriting(self) -> None:
        from src.ingestion.embedder import Embedder

        instance = Embedder.__new__(Embedder)
        assert isinstance(instance, EmbedderProtocol)
        assert EmbedderProtocol not in Embedder.__mro__  # structural, not nominal

    def test_gemini_embedder_satisfies_protocol_without_inheriting(self) -> None:
        instance = GeminiEmbedder.__new__(GeminiEmbedder)
        assert isinstance(instance, EmbedderProtocol)
        assert EmbedderProtocol not in GeminiEmbedder.__mro__
