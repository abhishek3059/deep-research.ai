"""Sliding-window conversation memory for multi-turn interactions."""

from __future__ import annotations


class ConversationMemory:
    """Maintains a fixed-size window of recent messages.

    Older messages are discarded when the window is exceeded.
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize memory.

        Args:
            window_size: Maximum number of messages to retain.
        """
        self._window_size = window_size
        self._messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Append a message, evicting the oldest if the window is full.

        Args:
            role: Message role (``"system"``, ``"user"``, ``"assistant"``).
            content: Message body.
        """
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._window_size:
            self._messages = self._messages[-self._window_size :]

    def get_messages(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all stored messages."""
        self._messages.clear()
