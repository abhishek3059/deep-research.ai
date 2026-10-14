"""Research chat interface page (full agent pipeline)."""

from __future__ import annotations

from typing import Any, cast

import httpx
import streamlit as st

from src.ui.components.chat import display_message, display_sources

API_BASE = "http://localhost:8000"


def _send_query(query: str) -> dict[str, Any] | None:
    """Send a query to the agent API and return the response."""
    try:
        response = httpx.post(
            f"{API_BASE}/api/v1/query",
            json={"query": query},
            timeout=120.0,
        )
        if response.status_code == 200:
            return cast("dict[str, Any] | None", response.json())
        elif response.status_code == 404:
            st.warning("No relevant documents found. Try ingesting some files first.")
            return None
        elif response.status_code == 422:
            detail = response.json().get("detail", "Query rejected.")
            st.warning(f"Query rejected by safety guard: {detail}")
            return None
        else:
            st.error(f"API error: HTTP {response.status_code}")
            return None
    except httpx.ConnectError:
        st.error("Cannot connect to API server. Is it running on port 8000?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _display_agent_trace(result: dict[str, Any]) -> None:
    """Show the critique-loop trace behind an answer."""
    iterations = result.get("iterations", 0)
    revised = result.get("revised", False)
    passed = result.get("critique_passed")
    summary = result.get("critique_summary", "")
    with st.expander(
        f"Agent trace — {iterations} iteration(s){', revised' if revised else ''}",
        expanded=False,
    ):
        if passed is True:
            st.success("Critique: PASS")
        elif passed is False:
            st.warning("Critique: FAIL — delivered best available answer")
        else:
            st.caption("Critique: not recorded for this answer")
        if summary:
            st.markdown(f"_{summary}_")


def render() -> None:
    """Render the research chat page."""
    st.title("💬 Research Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])
        if msg.get("sources"):
            display_sources(msg["sources"])
        if msg["role"] == "assistant" and msg.get("trace"):
            _display_agent_trace(msg["trace"])

    if query := st.chat_input("Ask a research question..."):
        display_message("user", query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Researching (retrieve → critique → revise)..."):
            result = _send_query(query)

        if result:
            answer = str(result.get("answer", ""))
            raw_sources = result.get("sources", [])
            sources = [str(s) for s in raw_sources] if isinstance(raw_sources, list) else []
            display_message("assistant", answer)
            display_sources(sources)
            _display_agent_trace(result)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "trace": {
                        "iterations": result.get("iterations", 0),
                        "revised": result.get("revised", False),
                        "critique_passed": result.get("critique_passed"),
                        "critique_summary": result.get("critique_summary", ""),
                    },
                }
            )
        else:
            display_message("assistant", "I couldn't find an answer. Please try rephrasing.")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I couldn't find an answer. Please try rephrasing.",
                }
            )


if __name__ == "__main__":
    render()
