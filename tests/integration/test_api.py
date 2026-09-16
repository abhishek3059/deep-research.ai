"""Integration tests for the API layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@dataclass
class MockMetadata:
    source: str = "test.pdf"
    source_type: str = "pdf"
    page: int = 1
    section: str | None = None
    ingested_at: str = "2024-01-01T00:00:00Z"
    content_hash: str = "abc123"


@dataclass
class MockSearchResult:
    id: str = "1"
    text: str = "Test document content"
    metadata: MockMetadata = field(default_factory=MockMetadata)
    score: float = 0.9


@dataclass
class MockRetrievalMeta:
    strategy: str = "dense"
    dense_results: int = 1
    sparse_results: int = 0
    reranked: bool = False
    latency_ms: float = 10.0


@dataclass
class MockRetrievalResult:
    query: str = "test query"
    expanded_queries: list[str] = field(default_factory=lambda: ["test query"])
    results: list[MockSearchResult] = field(
        default_factory=lambda: [MockSearchResult()]
    )
    retrieval_metadata: MockRetrievalMeta = field(default_factory=MockRetrievalMeta)


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint returns status ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_root_endpoint(client: TestClient) -> None:
    """Root endpoint returns service name."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "DeepResearch AI"


def test_query_endpoint(client: TestClient) -> None:
    """Query endpoint returns answer and sources when retrieval is mocked."""
    mock_result = MockRetrievalResult()
    mock_gen_result = {
        "answer": "Test document content",
        "sources": [{"source": "test.pdf", "source_type": "pdf", "id": "1", "score": "0.9", "reference": "[Source 1]"}],
        "query": "test query",
    }

    with (
        patch("src.api.routes.query.get_store"),
        patch("src.api.routes.query.RetrievalPipeline") as mock_cls,
        patch("src.api.routes.query.GenerationPipeline") as mock_gen_cls,
    ):
        mock_retrieval = AsyncMock()
        mock_retrieval.retrieve = AsyncMock(return_value=mock_result)
        mock_cls.return_value = mock_retrieval

        mock_gen = AsyncMock()
        mock_gen.generate_answer = AsyncMock(return_value=mock_gen_result)
        mock_gen_cls.return_value = mock_gen

        response = client.post(
            "/api/v1/query",
            json={"query": "test query"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "Test document content" in data["answer"]
    assert data["sources"] == ["test.pdf"]


def test_query_empty_query(client: TestClient) -> None:
    """Query endpoint rejects empty query strings."""
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400
