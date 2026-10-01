"""Input validation guards (middleware around agent input).

Runs two *enforcing* checks on an incoming user query:

1. Injection detection — prompt-injection phrases such as
   "ignore previous instructions" or "you are now".
2. Query length validation — rejects empty queries and queries longer
   than ``max_length`` characters.

Topic relevance is *advisory only* (see :meth:`InputGuard._check_topic`):
a pre-retrieval keyword gate cannot judge whether a query relates to the
user's documents, and hard-rejecting on vocabulary mismatch destroys trust.
Relevance is decided post-retrieval by the coverage floor in
:mod:`src.retrieval.pipeline`, which measures against the real corpus.
"""

from __future__ import annotations

import re

import structlog

from src.guardrails.models import GuardResult, Violation

logger = structlog.get_logger(__name__)

# Case-insensitive prompt-injection signatures.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+.*instructions", re.IGNORECASE),
    re.compile(r"forget\s+(your|all)\s+.*(instructions|rules)", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(have|had)\s+no\s+(rules|restrictions)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"bypass\s+(your|the)\s+(safety|guardrails?|filters?|restrictions?)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|instructions)", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
)

# Small-talk / non-research intents that are out of scope for the platform.
# NOTE: keyword heuristic, not a classifier — a research question that merely
# mentions one of these words (e.g. "weather patterns in climate papers") can
# be flagged; scoping research queries with topic_keywords avoids that.
_OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bweather\b", re.IGNORECASE),
    re.compile(r"\bsports?\s+scores?\b", re.IGNORECASE),
    re.compile(r"\blottery\b", re.IGNORECASE),
    re.compile(r"\bhoroscope\b", re.IGNORECASE),
    re.compile(r"\bcelebrity\s+gossip\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+a\s+joke\b", re.IGNORECASE),
    re.compile(r"\brecipe\b", re.IGNORECASE),
    re.compile(r"\bwhat('s| is) the time\b", re.IGNORECASE),
)


class InputGuard:
    """Validate an incoming user query before it reaches any agent."""

    def __init__(self, max_length: int = 10000, topic_keywords: list[str] | None = None) -> None:
        """Configure the input guard.

        Args:
            max_length: Maximum accepted query length in characters.
            topic_keywords: Optional allow-list; when provided, the query
                must contain at least one keyword (case-insensitive).
        """
        self._max_length = max_length
        self._topic_keywords = [k.lower() for k in topic_keywords] if topic_keywords else []

    async def validate(self, query: str) -> GuardResult:
        """Validate input query against injection and length rules.

        Topic relevance is advisory (logged, never a violation): the gate
        checks safety and shape, while the retrieval coverage floor decides
        relevance against the actual corpus.

        Args:
            query: Raw user query.

        Returns:
            GuardResult with ``passed`` True only when no rule fires.
            ``validated_output`` echoes the query on pass, else None.
            ``action_taken`` is "pass", or "reject" when any violation is
            high/critical, otherwise "reask".
        """
        violations: list[Violation] = []
        violations.extend(self._check_length(query))
        violations.extend(self._check_injection(query))
        violations.extend(self._check_topic(query))

        passed = not violations
        if passed:
            action = "pass"
        elif any(v.severity in ("high", "critical") for v in violations):
            action = "reject"
        else:
            action = "reask"

        result = GuardResult(
            passed=passed,
            original_input=query,
            validated_output=query if passed else None,
            violations=violations,
            action_taken=action,
        )
        logger.info(
            "Input validated",
            passed=passed,
            action=action,
            violations=len(violations),
        )
        return result

    def _check_length(self, query: str) -> list[Violation]:
        """Reject empty queries and queries exceeding max_length."""
        if not query.strip():
            return [
                Violation(
                    guard_name="InputGuard",
                    severity="high",
                    description="Query is empty.",
                    span=None,
                )
            ]
        if len(query) > self._max_length:
            return [
                Violation(
                    guard_name="InputGuard",
                    severity="high",
                    description=(f"Query length {len(query)} exceeds maximum {self._max_length}."),
                    span=(self._max_length, len(query)),
                )
            ]
        return []

    def _check_injection(self, query: str) -> list[Violation]:
        """Flag known prompt-injection phrases with character spans."""
        violations: list[Violation] = []
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(query)
            if match:
                violations.append(
                    Violation(
                        guard_name="InputGuard",
                        severity="critical",
                        description=(f"Possible prompt injection: matched '{match.group(0)}'."),
                        span=(match.start(), match.end()),
                    )
                )
        return violations

    def _check_topic(self, query: str) -> list[Violation]:
        """Advisory topic signal — never blocks the query.

        Logs (but does not report) off-topic-looking queries and
        topic-keyword misses. Rationale: a keyword gate cannot tell whether
        a paraphrased question relates to the user's documents; false
        rejects cost users, while a wasted retrieval costs cents. Relevance
        is enforced post-retrieval by the coverage floor instead.

        Always returns an empty list so ``passed`` is unaffected.
        """
        if not query.strip():
            return []  # Length check already reports empty queries.
        lowered = query.lower()
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(query):
                logger.warning(
                    "Query looks off-topic (advisory only, not blocking)",
                    query=query[:120],
                )
                return []
        if self._topic_keywords and not any(k in lowered for k in self._topic_keywords):
            logger.warning(
                "Query matches no topic keyword (advisory only, not blocking)",
                query=query[:120],
            )
        return []
