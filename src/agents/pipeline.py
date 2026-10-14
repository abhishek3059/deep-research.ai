"""Agent pipeline — composition root for the full query path.

Single place that wires the deployed ``POST /api/v1/query`` path:

```
InputGuard (graph intake) -> RetrievalPipeline -> GenerationPipeline
    -> SelfCritiqueAgent (retrieve -> critique -> revise, max 3)
    -> OutputGuard (graph deliver) -> AgentState
```

Guards live in the LangGraph nodes (``intake_node`` / ``deliver_node``),
never in this module — invariant #5 holds. Callers depend on this class,
not on the individual pieces, so the endpoint stays a thin adapter.
"""

from __future__ import annotations

import structlog

from src.agents.generation import GenerationPipeline
from src.agents.graph import SelfCritiqueGraph, build_graph
from src.agents.llm_provider import LLMProvider
from src.agents.self_critique import SelfCritiqueAgent
from src.agents.state import AgentState
from src.config.constants import MAX_ITERATIONS
from src.config.settings import settings
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstore.manager import get_store

logger = structlog.get_logger(__name__)


class AgentPipeline:
    """Run the full retrieve -> critique -> deliver loop for one query."""

    def __init__(self, graph: SelfCritiqueGraph) -> None:
        """Bind an already-compiled graph.

        Args:
            graph: Compiled LangGraph self-critique machine (owns guards).
        """
        self._graph = graph

    async def run(self, query: str, max_iterations: int | None = None) -> AgentState:
        """Execute intake -> research -> critique -> revise -> deliver.

        Args:
            query: User question (non-empty; blank is rejected by the caller).
            max_iterations: Overrides the graph loop guard when provided.

        Returns:
            Terminal :class:`AgentState` (``status == "deliver"``).
        """
        final = await self._graph.run(query, max_iterations=max_iterations)
        logger.info(
            "Agent pipeline delivered",
            query=query[:80],
            iterations=final.iteration_count,
            no_coverage=final.no_coverage,
        )
        return final


def build_agent_pipeline(
    max_iterations: int = MAX_ITERATIONS,
) -> AgentPipeline:
    """Build the production agent pipeline from application settings.

    Provider selection is config-only: the embedder comes from
    ``build_embedder()`` (``EMBEDDING_PROVIDER``) and the chat model from
    ``LLMProvider`` (``LLM_MODEL``), so switching to Gemini needs only env
    vars (ADR-012). Retrieval legs stay minimal (dense-only, no expansion)
    because the critique loop itself re-retrieves on context gaps.

    Args:
        max_iterations: Loop guard for the critique cycle.

    Returns:
        A ready-to-run :class:`AgentPipeline`.
    """
    llm = LLMProvider()
    retrieval = RetrievalPipeline(
        vector_store=get_store(),
        enable_sparse=False,
        enable_rerank=False,
        enable_multi_query=False,
        min_coverage=settings.min_coverage,
    )
    generation = GenerationPipeline(llm_provider=llm)
    agent = SelfCritiqueAgent(
        retrieval_pipeline=retrieval,
        generation_pipeline=generation,
        llm_provider=llm,
        max_iterations=max_iterations,
        top_k=settings.top_k,
    )
    return AgentPipeline(graph=build_graph(agent, max_iterations=max_iterations))
