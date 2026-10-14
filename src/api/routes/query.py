"""Query endpoint — full agent pipeline (retrieve -> critique -> deliver)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.exceptions import AgentError, AgentGraphError
from src.agents.pipeline import AgentPipeline, build_agent_pipeline

logger = structlog.get_logger(__name__)

router = APIRouter()


# Module-level singleton: the graph compiles once (LangGraph + model clients)
# and is stateless across requests, so it is built once and reused. Tests reset
# it by assigning ``None`` (see ``tests/integration/test_api.py``).
_agent: AgentPipeline | None = None


def _get_agent() -> AgentPipeline:
    """Return the shared agent pipeline, building it on first call."""
    global _agent  # noqa: PLW0603 - intentional process-wide singleton
    if _agent is None:
        _agent = build_agent_pipeline()
    return _agent


def reset_agent() -> None:
    """Reset the singleton (useful in tests)."""
    global _agent  # noqa: PLW0603 - intentional process-wide singleton
    _agent = None


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    iterations: int = 0
    revised: bool = False
    critique_passed: bool | None = None
    critique_summary: str = ""


@router.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Run the full agent loop and return a cited, critiqued answer.

    Path: InputGuard (intake) -> retrieve -> generate -> critique -> revise
    (max 3) -> OutputGuard (deliver). 404 when the corpus does not cover the
    query (coverage floor); 422 when the input guard rejects the query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        final = await _get_agent().run(request.query.strip())
    except AgentGraphError as exc:
        # Input-guard rejections and unroutable states land here.
        logger.warning("Agent graph rejected query", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AgentError as exc:
        logger.error("Agent pipeline failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Agent pipeline failed") from exc

    answer = final.current_answer
    if final.no_coverage or not answer.strip():
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found for the query",
        )

    sources = [s.get("source", "") for s in final.sources if s.get("source")]
    critique = final.critique_result
    revised = bool(final.revised_answer) and final.revised_answer != final.initial_answer
    return QueryResponse(
        answer=answer,
        sources=sources,
        iterations=final.iteration_count,
        revised=revised,
        critique_passed=critique.passed if critique is not None else None,
        critique_summary=critique.summary if critique is not None else "",
    )
