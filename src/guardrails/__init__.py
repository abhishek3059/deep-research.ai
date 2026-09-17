"""Public API for the guardrails module.

Guards are middleware: they wrap agent I/O without modifying agent logic.
An agent must work identically with guards enabled or disabled
(AGENTS.md invariant #5).
"""

from src.guardrails.input_guards import InputGuard
from src.guardrails.models import GuardResult, Violation
from src.guardrails.output_guards import OutputGuard
from src.guardrails.validators import HallucinationGuard, grounding_score

__all__ = [
    "GuardResult",
    "HallucinationGuard",
    "InputGuard",
    "OutputGuard",
    "Violation",
    "grounding_score",
]
