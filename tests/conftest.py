"""Shared pytest fixtures.

The suite must be **hermetic**: results cannot depend on a developer's local
``.env``. This was a real defect, found while adding a second embedding provider
(ADR-012): ``CriticAgent()`` constructs a CrewAI ``Agent``, which eagerly builds
an LLM from ``os.environ``. With a real ``.env`` present, all 22 crew tests failed
with ``Missing credentials`` — even though those tests only exercise pure parsing
logic and never call the model.

The fix neutralises provider credentials for the duration of the session. Dummy
values are sufficient because construction validates presence, not validity, and
no test performs a network call.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

#: Env vars read directly by third-party SDKs (not through ``Settings``).
#: Values must be plausible enough for object construction to succeed.
_DUMMY_LLM_ENV = {
    "LLM_MODEL": "gpt-4o-mini",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "ANTHROPIC_API_KEY": "test-not-a-real-key",
    "GEMINI_API_KEY": "test-not-a-real-key",
    "AZURE_API_KEY": "test-not-a-real-key",
    "AZURE_API_BASE": "https://example.invalid",
}


@pytest.fixture(autouse=True, scope="session")
def hermetic_llm_env() -> Iterator[None]:
    """Neutralise provider env vars so local ``.env`` cannot affect results.

    Overrides rather than deletes, because CrewAI's fallback path still attempts
    to construct a client even when ``LLM_MODEL`` is absent. Restores the prior
    values afterwards so interactive debugging is unaffected.
    """
    saved = {key: os.environ.get(key) for key in _DUMMY_LLM_ENV}
    os.environ.update(_DUMMY_LLM_ENV)
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
