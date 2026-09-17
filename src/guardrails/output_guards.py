"""Output validation guards (middleware around agent output).

Runs three checks on a generated answer:

1. Structured response format — rejects empty answers and answers that are
   too short or lack sentence/paragraph structure.
2. Citation validation — when ``require_citations`` is set and sources are
   provided, verifies the answer references them (``[1]`` markers,
   "according to ...", or a source title/text snippet). Also flags answers
   produced with no sources at all.
3. Toxicity detection — flags profanity/inappropriate language.
"""

from __future__ import annotations

import re

import structlog

from src.guardrails.models import GuardResult, Violation

logger = structlog.get_logger(__name__)

_MIN_ANSWER_CHARS = 20

_CITATION_MARKER_RE = re.compile(
    r"\[\d+\]|\(\d+\)|\bsources?\s*\d+\b|according\s+to\b|\bas\s+cited\b|\breferences?\b",
    re.IGNORECASE,
)

_STRUCTURE_MARKER_RE = re.compile(r"[.!?\n]")

# Minimal profanity/inappropriate-language lexicon (word-boundary matched).
_PROFANITY_WORDS: frozenset[str] = frozenset(
    {
        "fuck",
        "fucking",
        "shit",
        "bitch",
        "bastard",
        "asshole",
        "dick",
        "pussy",
        "whore",
        "slut",
        "cunt",
        "faggot",
        "retard",
        "kill yourself",
        "kys",
    }
)
_PROFANITY_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in sorted(_PROFANITY_WORDS)
)


class OutputGuard:
    """Validate an agent answer before it is returned to the user."""

    def __init__(self, require_citations: bool = True) -> None:
        """Configure the output guard.

        Args:
            require_citations: When True, answers produced without sources
                are flagged, and answers with sources must reference them.
        """
        self._require_citations = require_citations

    async def validate(
        self, answer: str, sources: list[dict[str, str]] | None = None
    ) -> GuardResult:
        """Validate agent output against format, citation, and safety rules.

        Args:
            answer: Generated answer text.
            sources: Optional retrieved sources as dicts (e.g.
                ``{"title": ..., "text": ...}``).

        Returns:
            GuardResult with ``passed`` True only when no rule fires.
            ``validated_output`` echoes the answer on pass, else None.
            ``action_taken`` is "pass", or "reject" when any violation is
            high/critical, otherwise "reask".
        """
        violations: list[Violation] = []
        violations.extend(self._check_format(answer))
        violations.extend(self._check_citations(answer, sources or []))
        violations.extend(self._check_toxicity(answer))

        passed = not violations
        if passed:
            action = "pass"
        elif any(v.severity in ("high", "critical") for v in violations):
            action = "reject"
        else:
            action = "reask"

        result = GuardResult(
            passed=passed,
            original_input=answer,
            validated_output=answer if passed else None,
            violations=violations,
            action_taken=action,
        )
        logger.info(
            "Output validated",
            passed=passed,
            action=action,
            violations=len(violations),
        )
        return result

    def _check_format(self, answer: str) -> list[Violation]:
        """Ensure the answer is non-empty and has sentence structure."""
        if not answer.strip():
            return [
                Violation(
                    guard_name="OutputGuard",
                    severity="high",
                    description="Answer is empty.",
                    span=None,
                )
            ]
        if len(answer.strip()) < _MIN_ANSWER_CHARS:
            return [
                Violation(
                    guard_name="OutputGuard",
                    severity="medium",
                    description="Answer is too short to be a structured response.",
                    span=(0, len(answer)),
                )
            ]
        if not _STRUCTURE_MARKER_RE.search(answer):
            return [
                Violation(
                    guard_name="OutputGuard",
                    severity="medium",
                    description="Answer lacks sentence or paragraph structure.",
                    span=None,
                )
            ]
        return []

    def _check_citations(self, answer: str, sources: list[dict[str, str]]) -> list[Violation]:
        """Verify the answer references its sources when required."""
        if not answer.strip():
            return []  # Format check already reports empty answers.
        if not sources:
            if self._require_citations:
                return [
                    Violation(
                        guard_name="OutputGuard",
                        severity="medium",
                        description="Answer produced without any sources.",
                        span=None,
                    )
                ]
            return []
        if _CITATION_MARKER_RE.search(answer):
            return []
        lowered = answer.lower()
        for source in sources:
            for key in ("title", "text", "content", "source"):
                value = source.get(key, "")
                snippet = value.strip().lower()
                # Match on a meaningful snippet so short titles still count
                # while avoiding single-word false positives.
                if len(snippet) >= 12 and snippet[:60] in lowered:
                    return []
                if len(snippet) >= 4 and key == "title" and snippet in lowered:
                    return []
        return [
            Violation(
                guard_name="OutputGuard",
                severity="medium",
                description="Answer does not reference any provided source.",
                span=None,
            )
        ]

    def _check_toxicity(self, answer: str) -> list[Violation]:
        """Flag profanity or inappropriate language with spans."""
        violations: list[Violation] = []
        for pattern in _PROFANITY_RES:
            match = pattern.search(answer)
            if match:
                violations.append(
                    Violation(
                        guard_name="OutputGuard",
                        severity="high",
                        description="Answer contains inappropriate language.",
                        span=(match.start(), match.end()),
                    )
                )
        return violations
