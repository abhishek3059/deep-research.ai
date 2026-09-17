"""LangGraph ``StateGraph`` for the Phase 2 self-critique agent (ADR-004).

Flow: ``INTAKE -> RESEARCH -> CRITIQUE -> {research, revise, deliver}`` with a
loop guard bounded by :data:`~src.config.constants.MAX_ITERATIONS`.

Week 1 of ADR-004 replaces the hand-rolled async state machine with LangGraph's
:class:`~langgraph.graph.StateGraph`.  Node logic is unchanged — it still lives
in :class:`~src.agents.self_critique.SelfCritiqueAgent` — so this module owns
control flow only: node registration, the conditional edge out of CRITIQUE, and
the iteration loop guard.

:class:`SelfCritiqueGraph` keeps the pre-LangGraph public surface (``build_graph``
returning an object with async ``run``) so existing callers keep working, while
new callers can drive the compiled graph directly via
:meth:`SelfCritiqueGraph.ainvoke`.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import structlog
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.exceptions import AgentGraphError, AgentLoopGuardError
from src.agents.self_critique import SelfCritiqueAgent
from src.agents.state import AgentState
from src.config.constants import MAX_ITERATIONS
from src.guardrails.input_guards import InputGuard
from src.guardrails.output_guards import OutputGuard

logger = structlog.get_logger(__name__)

# Allowed edges. ``critique`` branches to research / revise / deliver.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "intake": ("research",),
    "research": ("critique",),
    "critique": ("research", "revise", "deliver"),
    "revise": ("critique",),
    "deliver": (),
}

# Registered node names, in declaration order.
NODES: tuple[str, ...] = ("intake", "research", "critique", "revise", "deliver")

# Branch labels returned by ``route_after_critique``.
CRITIQUE_BRANCHES: tuple[str, ...] = ("research", "revise", "deliver")

# Recursion-limit floor (LangGraph super-steps).  One critique iteration costs
# at most four super-steps (research, critique, revise, critique); the floor
# keeps headroom for the intake -> critique -> deliver spine when
# ``max_iterations`` is tiny.
MIN_RECURSION_LIMIT = 25

# ``ContextT`` is unused by this graph, so it is pinned to ``None``.
SelfCritiqueStateGraph: TypeAlias = StateGraph[AgentState, None, AgentState, AgentState]
CompiledSelfCritiqueGraph: TypeAlias = CompiledStateGraph[
    AgentState, None, AgentState, AgentState
]


# ---------------------------------------------------------------------------
# Graph-owned nodes + router (pure functions, no instance state needed)
# ---------------------------------------------------------------------------

async def intake_node(state: AgentState) -> AgentState:
    """Validate the query and advance to RESEARCH."""
    if not state.query or not state.query.strip():
        raise AgentGraphError("Intake received an empty query")
    
    # Input guard: validate query safety
    input_guard = InputGuard()
    guard_result = await input_guard.validate(state.query)
    if not guard_result.passed:
        logger.warning("Input guard rejected query",
            violations=[v.description for v in guard_result.violations])
        raise AgentGraphError(
            f"Input rejected: {guard_result.violations[0].description}"
        )
    
    state.query = state.query.strip()
    state.status = "research"
    return state


async def deliver_node(state: AgentState) -> AgentState:
    """Terminal node — validate output and mark delivered."""
    # Output guard: validate answer quality
    output_guard = OutputGuard()
    answer = state.current_answer
    sources = state.sources if state.sources else None
    guard_result = await output_guard.validate(answer, sources)
    if not guard_result.passed:
        logger.warning("Output guard flagged issues",
            violations=[v.description for v in guard_result.violations])
    
    state.status = "deliver"
    return state


def route_after_critique(state: AgentState) -> str:
    """Pick the conditional edge out of CRITIQUE.

    Args:
        state: State produced by the agent's CRITIQUE node.

    Returns:
        One of :data:`CRITIQUE_BRANCHES`: ``"research"`` (context gap),
        ``"revise"`` (answer needs rework), or ``"deliver"`` (passed or the
        iteration budget is spent).

    Raises:
        AgentGraphError: If CRITIQUE left an unroutable status behind.
    """
    if state.status == "deliver":
        return "deliver"
    if state.iteration_count >= state.max_iterations:
        # Loop guard: budget spent, ship the best answer we have.
        return "deliver"
    if state.status in {"research", "revise"}:
        return state.status
    raise AgentGraphError(f"Critique produced an unroutable status: {state.status!r}")


def build_state_graph(agent: SelfCritiqueAgent) -> SelfCritiqueStateGraph:
    """Construct the LangGraph topology for the self-critique loop.

    Args:
        agent: Provides the research/critique/revise node implementations.

    Returns:
        An uncompiled :class:`~langgraph.graph.StateGraph`; call
        ``.compile()`` before invoking.
    """
    graph: SelfCritiqueStateGraph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("research", agent.research)
    graph.add_node("critique", agent.critique)
    graph.add_node("revise", agent.revise)
    graph.add_node("deliver", deliver_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "critique")
    graph.add_edge("revise", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {branch: branch for branch in CRITIQUE_BRANCHES},
    )
    graph.add_edge("deliver", END)
    return graph


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper
# ---------------------------------------------------------------------------

class SelfCritiqueGraph:
    """Compiled LangGraph state machine wrapping a :class:`SelfCritiqueAgent`.

    The graph owns control flow and the loop guard; the agent owns node logic.
    ``compile()`` is retained for API parity with the pre-LangGraph class and
    returns ``self`` — the underlying LangGraph is compiled in ``__init__``.
    """

    NODES: tuple[str, ...] = NODES

    def __init__(
        self,
        agent: SelfCritiqueAgent,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        """Compile the graph bound to *agent*.

        Args:
            agent: Provides research/critique/revise node implementations.
            max_iterations: Default loop guard for :meth:`run`.
        """
        self._agent = agent
        self._max_iterations = max_iterations
        self._state_graph = build_state_graph(agent)
        self._compiled: CompiledSelfCritiqueGraph = self._state_graph.compile()

    # -- properties -----------------------------------------------------------

    @property
    def nodes(self) -> tuple[str, ...]:
        """Registered node names, in declaration order."""
        return self.NODES

    @property
    def state_graph(self) -> SelfCritiqueStateGraph:
        """The uncompiled LangGraph topology (nodes + edges)."""
        return self._state_graph

    @property
    def compiled(self) -> CompiledSelfCritiqueGraph:
        """The compiled LangGraph runnable."""
        return self._compiled

    # -- lifecycle ------------------------------------------------------------

    def compile(self) -> SelfCritiqueGraph:
        """Return the ready-to-run graph (already compiled in ``__init__``)."""
        return self

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> AgentState:
        """Execute the graph until DELIVER.

        Args:
            query: User question.
            max_iterations: Overrides the instance guard when provided.

        Returns:
            Terminal :class:`AgentState` (``status == "deliver"``).
        """
        state = AgentState(
            query=query,
            max_iterations=max_iterations or self._max_iterations,
        )
        final = await self.ainvoke(state)

        logger.info(
            "Graph delivered",
            query=final.query[:80],
            iterations=final.iteration_count,
        )
        return final

    async def ainvoke(self, state: AgentState | dict[str, Any]) -> AgentState:
        """Run the compiled LangGraph and return the terminal state.

        Args:
            state: Initial graph state (model or raw mapping).

        Returns:
            The terminal :class:`AgentState`.

        Raises:
            AgentLoopGuardError: If LangGraph's recursion limit is hit before
                the graph delivers (defense in depth behind the edge guard).
        """
        resolved = state if isinstance(state, AgentState) else AgentState.model_validate(state)
        config: dict[str, int] = {
            "recursion_limit": self._recursion_limit(resolved.max_iterations)
        }
        try:
            raw: object = await self._compiled.ainvoke(resolved, config)
        except GraphRecursionError as exc:
            raise AgentLoopGuardError(
                "LangGraph recursion limit reached before the graph delivered"
            ) from exc
        return self._coerce_state(raw)

    def route(self, state: AgentState) -> str:
        """Choose the next node, enforcing the iteration loop guard.

        Mirrors :func:`route_after_critique`; retained so callers can inspect
        routing decisions without executing the graph.
        """
        return route_after_critique(state)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _recursion_limit(max_iterations: int) -> int:
        """Convert an iteration budget into a LangGraph recursion limit."""
        return max(MIN_RECURSION_LIMIT, max_iterations * 4 + 10)

    @staticmethod
    def _coerce_state(raw: object) -> AgentState:
        """Normalize LangGraph's dict output back into an :class:`AgentState`."""
        if isinstance(raw, AgentState):
            return raw
        if isinstance(raw, dict):
            return AgentState.model_validate(raw)
        raise AgentGraphError(f"Unexpected graph output type: {type(raw).__name__}")


def build_graph(
    agent: SelfCritiqueAgent,
    max_iterations: int = MAX_ITERATIONS,
) -> SelfCritiqueGraph:
    """Build (and compile) the self-critique LangGraph state machine.

    Args:
        agent: Node provider.
        max_iterations: Loop guard.

    Returns:
        A compiled :class:`SelfCritiqueGraph`.
    """
    return SelfCritiqueGraph(agent, max_iterations=max_iterations).compile()
