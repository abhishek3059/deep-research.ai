"""Hallucination / grounding validators.

Compares the claims in an answer against the retrieved context using a
deterministic lexical-overlap scorer (no LLM calls): each answer sentence
counts as supported when at least half of its content tokens appear in the
combined context vocabulary. The grounding score is the share of supported
sentences (0-1); answers scoring below ``threshold`` are rejected.
"""

from __future__ import annotations

import re

import structlog

from src.guardrails.models import GuardResult, Violation

logger = structlog.get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    "a an the and or but if then else for of in on at to from with by is are was were "
    "be been being it its this that these those as what which who whom how why when "
    "where does do did can could should would will just very so than too not no "
    "we you they he she them his her our your their there here has have had has".split()
)


def _stem(word: str) -> str:
    """Strip common suffixes so plurals match their singulars."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    """Tokenize to lowercase stemmed content words minus stopwords."""
    return {_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    return [s.strip() for s in re.split(r"[.!?]+|\n+", text) if s.strip()]


def grounding_score(answer: str, context: list[str]) -> float:
    """Calculate the 0-1 grounding score of an answer against context.

    Args:
        answer: Generated answer text.
        context: Retrieved context passages.

    Returns:
        Share of answer sentences supported by the context vocabulary
        (0.0 when the answer or context is empty), rounded to 4 decimals.
    """
    sentences = _sentences(answer)
    if not sentences or not context:
        return 0.0
    vocab: set[str] = set()
    for passage in context:
        vocab |= _tokens(passage)
    if not vocab:
        return 0.0

    def _support_ratio(sentence_tokens: set[str]) -> float:
        if not sentence_tokens:
            return 0.0
        return len(sentence_tokens & vocab) / len(sentence_tokens)

    supported = sum(1 for s in sentences if _support_ratio(_tokens(s)) >= 0.5)
    return round(supported / len(sentences), 4)


class HallucinationGuard:
    """Validate that an answer is grounded in the provided context."""

    def __init__(self, threshold: float = 0.3) -> None:
        """Configure the hallucination guard.

        Args:
            threshold: Minimum grounding score (0-1) for acceptance.
        """
        self._threshold = threshold

    async def validate(self, answer: str, context: list[str]) -> GuardResult:
        """Validate answer is grounded in provided context.

        Args:
            answer: Generated answer text.
            context: Retrieved context passages the answer must ground to.

        Returns:
            GuardResult with ``passed`` True when the grounding score is
            at or above the threshold. ``validated_output`` echoes the
            answer on pass, else None. Below-threshold answers are
            rejected ("reject").
        """
        score = grounding_score(answer, context)
        violations: list[Violation] = []
        if score < self._threshold:
            violations.append(
                Violation(
                    guard_name="HallucinationGuard",
                    severity="high",
                    description=(
                        f"Answer grounding score {score} below threshold {self._threshold}."
                    ),
                    span=None,
                )
            )
        passed = not violations
        result = GuardResult(
            passed=passed,
            original_input=answer,
            validated_output=answer if passed else None,
            violations=violations,
            action_taken="pass" if passed else "reject",
        )
        logger.info(
            "Hallucination check complete",
            passed=passed,
            score=score,
            threshold=self._threshold,
        )
        return result
