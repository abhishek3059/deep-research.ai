"""Query endpoint for retrieval and generation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.generation import GenerationPipeline
from src.config.settings import settings
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.manager import get_store

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Run retrieval and return an answer with cited sources.

    Uses the retrieval pipeline with multi-query expansion and hybrid
    search.  The answer is assembled from retrieved chunks.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    store = get_store()

    retrieval = RetrievalPipeline(
        vector_store=store,
        enable_sparse=False,
        enable_rerank=False,
        enable_multi_query=False,
    )

    result = await retrieval.retrieve(request.query, top_k=settings.top_k)

    if not result.results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found for the query",
        )

    generator = GenerationPipeline()
    gen_result = await generator.generate_answer(request.query, result)

    sources = [s["source"] for s in gen_result["sources"]]
    return QueryResponse(answer=gen_result["answer"], sources=sources)