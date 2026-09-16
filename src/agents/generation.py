"""Generation pipeline — the final stage of the RAG system.

Takes a user query plus a :class:`RetrievalResult` from the retrieval module
and produces a cited answer via the configured LLM.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.agents.llm_provider import LLMProvider
from src.agents.memory import ConversationMemory
from src.agents.prompts import FORMAT_INSTRUCTIONS, RESEARCH_PROMPT
from src.retrieval.pipeline import RetrievalResult

logger = structlog.get_logger(__name__)


class GenerationPipeline:
    """Assemble context, call the LLM, and return a structured answer dict."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        """Wire dependencies.

        Args:
            llm_provider: LLM wrapper.  Defaults to a new :class:`LLMProvider`.
            memory: Conversation memory.  Defaults to a 5-message window.
        """
        self._llm = llm_provider or LLMProvider()
        self._memory = memory or ConversationMemory()

    async def generate_answer(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> dict[str, Any]:
        """Build context from retrieval results and generate an answer.

        Args:
            query: The user's original question.
            retrieval_result: Ranked results from the retrieval pipeline.

        Returns:
            Dict with keys ``answer``, ``sources``, ``query``.
        """
        context = self._build_context(retrieval_result)
        sources = self._extract_sources(retrieval_result)

        messages = self._build_messages(query, context)
        answer = await self._llm.generate(messages)

        self._memory.add_message("user", query)
        self._memory.add_message("assistant", answer)

        logger.info("Generation completed", query=query[:80], sources_count=len(sources))

        return {"answer": answer, "sources": sources, "query": query}

    # -- private --------------------------------------------------------------

    def _build_messages(self, query: str, context: str) -> list[dict[str, str]]:
        """Assemble the full message list for the LLM."""
        system_content = f"{RESEARCH_PROMPT}\n\n{FORMAT_INSTRUCTIONS}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(self._memory.get_messages())
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        )
        return messages

    @staticmethod
    def _build_context(result: RetrievalResult) -> str:
        """Format retrieved chunks into a numbered context string."""
        parts: list[str] = []
        for idx, sr in enumerate(result.results, 1):
            parts.append(f"[Source {idx}] {sr.text}")
        return "\n\n".join(parts) if parts else "No relevant context found."

    @staticmethod
    def _extract_sources(result: RetrievalResult) -> list[dict[str, str]]:
        """Pull lightweight source references from search results."""
        sources: list[dict[str, str]] = []
        for idx, sr in enumerate(result.results, 1):
            sources.append(
                {
                    "id": sr.id,
                    "source": sr.metadata.source,
                    "source_type": sr.metadata.source_type,
                    "score": str(sr.score),
                    "reference": f"[Source {idx}]",
                }
            )
        return sources
