"""Research chat interface page."""

from __future__ import annotations

import httpx
import streamlit as st

from src.ui.components.chat import display_message, display_sources

API_BASE = "http://localhost:8000"


def _send_query(query: str) -> dict[str, object] | None:
    """Send a query to the retrieval API and return the response."""
    try:
        response = httpx.post(
            f"{API_BASE}/api/v1/query",
            json={"query": query},
            timeout=60.0,
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.warning("No relevant documents found. Try ingesting some files first.")
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


def render() -> None:
    """Render the research chat page."""
    st.title("💬 Research Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])
        if msg.get("sources"):
            display_sources(msg["sources"])

    if query := st.chat_input("Ask a research question..."):
        display_message("user", query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Searching knowledge base..."):
            result = _send_query(query)

        if result:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            display_message("assistant", answer)
            display_sources(sources)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            display_message("assistant", "I couldn't find an answer. Please try rephrasing.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I couldn't find an answer. Please try rephrasing.",
            })


if __name__ == "__main__":
    render()
