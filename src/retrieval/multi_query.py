"""Multi-query expansion for improved recall."""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_PROMPT = (
    "You are a helpful research assistant. Generate {n} alternative phrasings "
    "of the user's question that capture different aspects. Return ONLY the "
    "rewritten queries, one per line, with no numbering or explanation.\n\n"
    "Original: {query}"
)


class MultiQueryExpander:
    """Generate query variations to broaden retrieval coverage."""

    def __init__(self, llm: object | None = None) -> None:
        """Initialise with an optional LLM callable.

        Args:
            llm: An object with a ``.invoke(prompt: str) -> str`` method
                 (e.g. a LangChain ChatModel). When *None* the expander
                 returns only the original query.
        """
        self._llm = llm

    def expand(self, query: str, num_queries: int = 3) -> list[str]:
        """Produce *num_queries* alternative phrasings.

        Args:
            query: The original user query.
            num_queries: How many rewrites to request (excluding the original).

        Returns:
            List containing the original query plus rewrites.
        """
        if self._llm is None:
            logger.debug("No LLM configured, returning original query only")
            return [query]

        prompt = _DEFAULT_PROMPT.format(n=num_queries, query=query)
        try:
            response = self._llm.invoke(prompt)
            lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
            # Drop any numbered prefixes like "1. " or "1) "
            cleaned = [
                line.split(".", 1)[-1].strip() if line[0].isdigit() else line for line in lines
            ]
            return [query] + cleaned[:num_queries]
        except Exception:
            logger.warning("Multi-query expansion failed, falling back to original")
            return [query]
