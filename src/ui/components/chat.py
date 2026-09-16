"""Chat display components for Streamlit."""

from __future__ import annotations

import streamlit as st


def display_message(role: str, content: str) -> None:
    """Display a single chat message.

    Args:
        role: Either "user" or "assistant".
        content: The message content to render.
    """
    with st.chat_message(role):
        st.markdown(content)


def display_sources(sources: list[str]) -> None:
    """Display source citations below an assistant response.

    Args:
        sources: List of source file paths or URLs referenced in the answer.
    """
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**[{i}]** `{source}`")
