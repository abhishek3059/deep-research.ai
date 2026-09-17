"""Shared dataclasses for the guardrails module.

Matches docs/CONTRACTS.md section 4.4 (Agents -> Guardrails).
Guards never mutate agent logic — they wrap agent I/O and report a
:class:`GuardResult` describing pass/fail, violations, and the action taken.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Violation:
    """A single rule violation found by a guard.

    Matches docs/CONTRACTS.md section 4.4.
    """

    guard_name: str  # e.g. "InputGuard", "OutputGuard", "HallucinationGuard"
    severity: str  # "low" | "medium" | "high" | "critical"
    description: str  # Human-readable explanation
    span: tuple[int, int] | None = None  # Character offsets if applicable


@dataclass
class GuardResult:
    """Outcome of a guard validation.

    Matches docs/CONTRACTS.md section 4.4.
    """

    passed: bool
    original_input: str
    validated_output: str | None  # None if rejected
    violations: list[Violation]
    action_taken: str  # "pass" | "fix" | "reject" | "reask"
