"""Pure-Python async state machine for the self-check review agent.

States: ``INTAKE -> RESEARCH -> REVIEW -> REVISE -> DELIVER`` with a
conditional edge from REVIEW back to RESEARCH (grounding gap) and a
REVIEW -> REVISE edge for answers that merely need reworking.

Why pure Python instead of LangGraph?  Same rationale as
:mod:`src.agents.graph` (ADR-003): LangGraph is not declared in
``pyproject.toml``, so using it would violate the dependency-discipline rule.
This module has no third-party dependencies beyond the project's own state
model.  Node logic lives in :class:`~src.agents.review_agent.ReviewAgent`; the
graph owns only control flow and the loop guard, mirroring LangGraph's
node/edge vocabulary so it can be swapped later.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog

from src.agents.exceptions import AgentGraphError, AgentLoopGuardError
from src.agents.review_agent import ReviewAgent
from src.agents.state import AgentState
from src.config.constants import MAX_ITERATIONS

logger = structlog.get_logger(__name__)

NodeFn = Callable[[AgentState], Awaitable[AgentState]]

REVIEW_STATUSES: tuple[str, ...] = (
    "intake",
    "research",
    "review",
    "revise",
    "deliver",
)

# Allowed edges.  ``review`` branches to research / revise / deliver.
REVIEW_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "intake": ("research",),
    "research": ("review",),
    "review": ("research", "revise", "deliver"),
    "revise": ("review",),
    "deliver": (),
}


class ReviewGraph:
    """Async finite state machine wrapping a :class:`ReviewAgent`.

    The graph owns control flow and the iteration loop guard; the agent owns
    node logic.  ``research_results`` accumulates across REVIEW -> RESEARCH
    loop-backs (see :meth:`AgentState.add_research_results`).
    """

    NODES: tuple[str, ...] = REVIEW_STATUSES

    def __init__(
        self,
        agent: ReviewAgent,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        """Bind the agent (node provider) and the loop guard.

        Args:
            agent: Provides research/review/revise node implementations.
            max_iterations: Hard cap on REVIEW iterations.
        """
        self._agent = agent
        self._max_iterations = max_iterations
        self._nodes: dict[str, NodeFn] = {
            "intake": self._intake,
            "research": agent.research,
            "review": agent.review,
            "revise": agent.revise,
            "deliver": self._deliver,
        }

    # -- lifecycle ------------------------------------------------------------

    def compile(self) -> ReviewGraph:
        """Return the ready-to-run graph.

        Present for LangGraph API parity — the pure-Python machine needs no
        separate compilation step.
        """
        return self

    @property
    def nodes(self) -> tuple[str, ...]:
        """Registered node names, in declaration order."""
        return self.NODES

    async def run(
        self,
        query: str,
        max_iterations: int | None = None,
    ) -> AgentState:
        """Execute the state machine until DELIVER.

        Args:
            query: User question.
            max_iterations: Overrides the instance guard when provided.

        Returns:
            Terminal :class:`AgentState` (``status == "deliver"``).

        Raises:
            AgentGraphError: On an unknown status or disallowed edge.
            AgentLoopGuardError: If the step guard is exceeded.
        """
        state = AgentState(
            query=query,
            max_iterations=max_iterations or self._max_iterations,
        )
        step_guard = state.max_iterations * len(self._nodes) + 10
        steps = 0

        while state.status != "deliver":
            next_status = self.route(state)
            node = self._nodes.get(next_status)
            if node is None:
                raise AgentGraphError(f"Unknown node: {next_status!r}")

            previous = state.status
            state = await node(state)
            self._assert_transition(previous, state.status)

            steps += 1
            if steps > step_guard:
                raise AgentLoopGuardError(
                    f"Graph exceeded {step_guard} steps without delivering"
                )

        logger.info(
            "Review graph delivered",
            query=state.query[:80],
            iterations=state.iteration_count,
            steps=steps,
        )
        return state

    def route(self, state: AgentState) -> str:
        """Choose the next node, enforcing the iteration loop guard.

        Args:
            state: Current state.

        Returns:
            Name of the node to execute next.
        """
        if state.status == "deliver":
            return "deliver"
        if (
            state.iteration_count >= state.max_iterations
            and state.status in {"research", "review", "revise"}
        ):
            # Loop guard: stop the REVIEW -> RESEARCH cycle and deliver.
            return "deliver"
        return state.status

    # -- nodes ----------------------------------------------------------------

    async def _intake(self, state: AgentState) -> AgentState:
        """Validate the query and advance to RESEARCH."""
        if not state.query or not state.query.strip():
            raise AgentGraphError("Intake received an empty query")
        state.query = state.query.strip()
        state.status = "research"
        return state

    async def _deliver(self, state: AgentState) -> AgentState:
        """Terminal node — mark the state delivered."""
        state.status = "deliver"
        return state

    # -- invariants -----------------------------------------------------------

    @staticmethod
    def _assert_transition(previous: str, current: str) -> None:
        """Raise if *previous -> current* is not a legal edge.

        ``deliver`` is a universal sink: the loop guard may force delivery from
        research/review/revise when the iteration budget is exhausted.
        """
        if current == "deliver":
            return
        allowed = REVIEW_TRANSITIONS.get(previous)
        if allowed is None:
            raise AgentGraphError(f"Unknown source node: {previous!r}")
        if current not in allowed:
            raise AgentGraphError(f"Illegal transition: {previous!r} -> {current!r}")


def build_review_graph(
    agent: ReviewAgent,
    max_iterations: int = MAX_ITERATIONS,
) -> ReviewGraph:
    """Build (and compile) the self-check review state machine.

    Args:
        agent: Node provider.
        max_iterations: Loop guard.

    Returns:
        A compiled :class:`ReviewGraph`.
    """
    return ReviewGraph(agent, max_iterations=max_iterations).compile()
