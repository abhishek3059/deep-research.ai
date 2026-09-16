"""Unified LLM provider abstraction.

Wraps langchain OpenAI and Anthropic chat models behind a single async
interface so callers can switch providers via configuration alone.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.config.settings import settings

logger = structlog.get_logger(__name__)

# Provider registry — lazy imports to avoid importing unused SDKs.
_CHAT_MODEL: dict[str, Any] | None = None


def _get_chat_model() -> dict[str, Any]:
    """Return a mapping of provider name -> chat model class (lazy)."""
    global _CHAT_MODEL  # noqa: PLW0603
    if _CHAT_MODEL is None:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI

        _CHAT_MODEL = {
            "openai": ChatOpenAI,
            "anthropic": ChatAnthropic,
        }
    return _CHAT_MODEL


class LLMProvider:
    """Thin async wrapper around langchain chat models.

    Defaults are pulled from :class:`Settings` when not explicitly provided.
    """

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        """Initialize the provider.

        Args:
            provider: ``"openai"`` or ``"anthropic"``.  Falls back to
                ``settings.llm_model`` heuristic (OpenAI for gpt-*,
                Anthropic otherwise).
            model: Model name.  Defaults to ``settings.llm_model``.
        """
        self._provider = provider or self._infer_provider(settings.llm_model)
        self._model = model or settings.llm_model
        self._llm = self._build_llm()

    # -- public ---------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """Send *messages* to the configured LLM and return the text response.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature.

        Returns:
            The assistant message content string.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))

        logger.debug(
            "LLM call",
            provider=self._provider,
            model=self._model,
            n_messages=len(lc_messages),
        )
        response = await self._llm.ainvoke(lc_messages)
        return response.content

    # -- private --------------------------------------------------------------

    @staticmethod
    def _infer_provider(model: str) -> str:
        """Heuristic: gpt-* → openai, everything else → anthropic."""
        if model.startswith("gpt") or model.startswith("o1"):
            return "openai"
        return "anthropic"

    def _build_llm(self) -> Any:
        """Instantiate the langchain chat model for the chosen provider."""
        models = _get_chat_model()
        cls = models[self._provider]
        kwargs: dict[str, Any] = {"model": self._model, "temperature": 0.3}
        if self._provider == "openai":
            kwargs["api_key"] = settings.openai_api_key or None
        elif self._provider == "anthropic":
            kwargs["api_key"] = settings.anthropic_api_key or None
        return cls(**kwargs)
