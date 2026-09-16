"""Exceptions for the agent orchestration module.

Kept in a leaf module (not ``__init__.py``) so :mod:`src.agents.graph` and
:mod:`src.agents.self_critique` can import them without a circular import.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent orchestration errors."""


class AgentGraphError(AgentError):
    """Raised when the agent state machine hits an invalid state or loop."""


class AgentLoopGuardError(AgentGraphError):
    """Raised when the graph exceeds its iteration guard."""
