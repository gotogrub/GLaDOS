"""Think tag filter and sentence splitter for LLM output streaming."""
from __future__ import annotations

from loguru import logger


class ThinkFilter:
    """Strips <think>...</think> and <thinking>...</thinking> blocks."""

    OPEN_TAGS = ("<think>", "<thinking>")
    CLOSE_TAGS = ("</think>", "</thinking>")

    def __init__(self) -> None:
        self._in_thinking = False
        self._buf: list[str] = []

    def feed(self, chunk: str) -> str:
        """Feed a chunk. Returns speakable text (may be empty if inside think block)."""
        text = chunk

        if not self._in_thinking:
            for tag in self.OPEN_TAGS:
                if tag in text:
                    parts = text.split(tag, 1)
                    text = parts[0]
                    self._in_thinking = True
                    self._buf.append(parts[1] if len(parts) > 1 else "")
                    break

        if self._in_thinking:
            combined = "".join(self._buf) + text
            for tag in self.CLOSE_TAGS:
                if tag in combined:
                    parts = combined.split(tag, 1)
                    if parts[0].strip():
                        logger.debug("Thinking: {}...", parts[0][:100])
                    self._buf.clear()
                    self._in_thinking = False
                    return parts[1] if len(parts) > 1 else ""
            self._buf.append(text)
            return ""

        return text

    def reset(self) -> None:
        self._in_thinking = False
        self._buf.clear()


class SentenceSplitter:
    """Accumulates tokens and splits on sentence-ending punctuation."""

    ENDINGS = frozenset({".", "!", "?", ":", ";", "?!", "\n"})

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def feed(self, token: str) -> str | None:
        """Feed a token. Returns complete sentence if boundary found, else None."""
        self._buffer.append(token)
        combined = "".join(self._buffer)
        if any(combined.rstrip().endswith(p) for p in self.ENDINGS):
            self._buffer.clear()
            stripped = combined.strip()
            return stripped if stripped else None
        return None

    def flush(self) -> str:
        """Flush remaining buffer."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return text
