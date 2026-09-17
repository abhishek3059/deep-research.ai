"""CrewAI agent definitions for DeepResearch AI.

Currently implements a single CriticAgent that reviews main agent output
for quality. Per ADR-004, we start with ONE agent to demonstrate the
pattern without over-engineering.
"""

from __future__ import annotations

import structlog
from crewai import Agent, Crew, Task

logger = structlog.get_logger(__name__)


class CriticAgent:
    """CrewAI agent that reviews research answers for quality.

    The CriticAgent evaluates answers against three dimensions:
    - Faithfulness: Is the answer grounded in the provided context?
    - Relevancy: Does the answer address the original query?
    - Completeness: Does the answer cover all aspects of the query?
    """

    def __init__(self) -> None:
        """Initialize the CriticAgent with a CrewAI Agent."""
        self._agent = Agent(
            role="Quality Critic",
            goal=(
                "Review research answers for accuracy, completeness, and "
                "grounding in provided context. Identify any hallucinations, "
                "missing information, or irrelevant content."
            ),
            backstory=(
                "You are an expert fact-checker and quality reviewer with "
                "deep experience in evaluating research outputs. You carefully "
                "compare answers against source materials and flag any issues "
                "with precision and clarity."
            ),
            verbose=True,
            allow_delegation=False,
        )

    async def review(
        self,
        answer: str,
        context: list[str],
        query: str,
    ) -> dict[str, object]:
        """Review the answer and provide structured feedback.

        Args:
            answer: The generated answer to review.
            context: The retrieved context used to generate the answer.
            query: The original user query.

        Returns:
            A dictionary with:
            - passed: bool indicating if the answer passes quality checks
            - faithfulness: float (0-1) score for context grounding
            - relevancy: float (0-1) score for query alignment
            - completeness: float (0-1) score for coverage
            - issues: list of specific issues found
            - summary: str with overall assessment
        """
        context_text = "\n\n".join(context) if context else "No context provided"

        review_task = Task(
            description=(
                f"Review this research answer for quality.\n\n"
                f"Original Query: {query}\n\n"
                f"Generated Answer:\n{answer}\n\n"
                f"Source Context:\n{context_text}\n\n"
                f"Evaluate the answer on these dimensions:\n"
                f"1. Faithfulness: Is the answer grounded in the context?\n"
                f"2. Relevancy: Does it address the query?\n"
                f"3. Completeness: Does it cover all aspects?\n\n"
                f"Provide scores (0-1) for each dimension and list specific issues."
            ),
            expected_output=(
                "A JSON-like structure with: passed (bool), faithfulness (float), "
                "relevancy (float), completeness (float), issues (list), summary (str)"
            ),
            agent=self._agent,
        )

        crew = Crew(
            agents=[self._agent],
            tasks=[review_task],
            verbose=True,
        )

        # Run the crew
        result = crew.kickoff()

        # Parse the result (CrewAI returns a string, we need to extract scores)
        review_result = self._parse_review(str(result), answer, context)

        logger.info(
            "CriticAgent review complete",
            passed=review_result["passed"],
            faithfulness=review_result["faithfulness"],
            relevancy=review_result["relevancy"],
            completeness=review_result["completeness"],
        )

        return review_result

    def _parse_review(
        self,
        raw_result: str,
        answer: str,
        context: list[str],
    ) -> dict[str, object]:
        """Parse CrewAI's raw output into structured review.

        Args:
            raw_result: Raw string output from CrewAI.
            answer: The original answer (for fallback scoring).
            context: The original context (for fallback scoring).

        Returns:
            Structured review dictionary.
        """
        # Simple heuristic parsing - look for scores in the output
        faithfulness = self._extract_score(raw_result, "faithfulness", answer, context)
        relevancy = self._extract_score(raw_result, "relevancy", answer, context)
        completeness = self._extract_score(raw_result, "completeness", answer, context)

        # Determine pass/fail (all scores must be >= 0.5)
        passed = faithfulness >= 0.5 and relevancy >= 0.5 and completeness >= 0.5

        # Extract issues from the raw result
        issues = []
        if "hallucination" in raw_result.lower():
            issues.append("Potential hallucination detected")
        if "irrelevant" in raw_result.lower():
            issues.append("Contains irrelevant information")
        if "incomplete" in raw_result.lower():
            issues.append("Answer may be incomplete")

        return {
            "passed": passed,
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "completeness": completeness,
            "issues": issues,
            "summary": raw_result[:500] if raw_result else "Review completed",
        }

    def _extract_score(
        self,
        text: str,
        dimension: str,
        answer: str,
        context: list[str],
    ) -> float:
        """Extract a score for a dimension from the review text.

        Args:
            text: The review text to search.
            dimension: The dimension name to look for.
            answer: The original answer (for fallback).
            context: The original context (for fallback).

        Returns:
            Score between 0.0 and 1.0.
        """
        import re

        # Try to find a score pattern like "faithfulness: 0.8" or "faithfulness score: 8/10"
        patterns = [
            rf"{dimension}[:\s]+(\d+\.?\d*)/10",
            rf"{dimension}[:\s]+(\d+\.?\d*)",
            rf"{dimension}[:\s]+(\d+\.?\d*)\s*%",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 if needed
                if score > 1:
                    score = score / 10 if score <= 10 else score / 100
                return min(max(score, 0.0), 1.0)

        # Fallback: simple keyword overlap scoring
        if not context:
            return 0.5  # No context to compare against

        answer_words = set(answer.lower().split())
        context_words = set(" ".join(context).lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words) / len(answer_words)
        return min(overlap, 1.0)
