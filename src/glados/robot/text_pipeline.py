"""Think tag filter, emotion parser, and sentence splitter for LLM output streaming."""
from __future__ import annotations

import re

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


class EmotionParser:
    """Extracts emotion tag from the start of LLM output.

    Supports two formats:
    - New: [emotion:sarcastic,0.8] — 12 emotions with intensity
    - Legacy: [SARCASM] — mapped to new system for backward compat

    Usage in streaming pipeline (sits between ThinkFilter and ChunkSplitter):
        parser = EmotionParser()
        for token in stream:
            text = parser.feed(token)
            if text:
                # pass to ChunkSplitter
        emotion = parser.emotion      # "sarcastic", "cold", etc.
        intensity = parser.intensity   # 0.0-1.0
    """

    EMOTIONS = frozenset({
        "sarcastic", "annoyed", "condescending", "curious",
        "disappointed", "menacing", "fake_pleasant", "bored",
        "angry", "amused", "contemplative", "cold",
    })

    # [emotion:sarcastic,0.8] or [emotion:sarcastic,high] or [emotion:sarcastic]
    _TAG_RE = re.compile(r"\s*\[emotion:([a-z_]+)(?:,([^\]]*))?\]\s*")
    # Legacy: [SARCASM]
    _OLD_TAG_RE = re.compile(r"\s*\[([A-Z_]+)\]\s*")
    _OLD_EMOTION_MAP = {
        "NEUTRAL": "cold", "SARCASM": "sarcastic", "ANGER": "angry",
        "CURIOSITY": "curious", "DISGUST": "annoyed", "AMUSEMENT": "amused",
        "SADNESS": "disappointed", "SURPRISE": "curious",
    }
    _MAX_BUFFER = 50  # give up after this many chars
    _DEFAULT_EMOTION = "sarcastic"

    def __init__(self) -> None:
        self._detected = False
        self._buffer: list[str] = []
        self.emotion: str = self._DEFAULT_EMOTION
        self.intensity: float = 0.5

    def feed(self, token: str) -> str:
        """Feed a token. Returns text to pass downstream (may be empty while buffering)."""
        if self._detected:
            return token

        self._buffer.append(token)
        combined = "".join(self._buffer)

        # Try new format: [emotion:name,intensity]
        m = self._TAG_RE.match(combined)
        if m:
            name = m.group(1)
            if name in self.EMOTIONS:
                self.emotion = name
                self.intensity = self._parse_intensity(m.group(2))
                self._detected = True
                logger.info("Emotion: {} ({:.1f})", self.emotion, self.intensity)
                return combined[m.end():]
            self._detected = True
            return combined

        # Try legacy format: [SARCASM]
        m_old = self._OLD_TAG_RE.match(combined)
        if m_old:
            old_name = m_old.group(1)
            if old_name in self._OLD_EMOTION_MAP:
                self.emotion = self._OLD_EMOTION_MAP[old_name]
                self._detected = True
                logger.info("Emotion (legacy): {} → {}", old_name, self.emotion)
                return combined[m_old.end():]
            self._detected = True
            return combined

        # Give up if: too much text, passed a `]`, or no `[` in sight
        stripped = combined.lstrip()
        if (len(combined) > self._MAX_BUFFER
                or "]" in combined
                or (len(stripped) > 3 and not stripped.startswith("["))):
            self._detected = True
            return combined

        return ""  # keep buffering

    @staticmethod
    def _parse_intensity(raw: str | None) -> float:
        if not raw:
            return 0.5
        raw = raw.strip().lower()
        try:
            return float(raw)
        except ValueError:
            word_map = {"low": 0.3, "medium": 0.5, "high": 0.8, "max": 1.0}
            return word_map.get(raw, 0.5)

    def flush(self) -> str:
        """Flush buffered text (call at end of stream)."""
        if self._detected:
            return ""
        self._detected = True
        combined = "".join(self._buffer)
        self._buffer.clear()
        return combined

    def reset(self) -> None:
        self._detected = False
        self._buffer.clear()
        self.emotion = self._DEFAULT_EMOTION
        self.intensity = 0.5


class ChunkSplitter:
    """Splits LLM output into TTS-friendly chunks.

    Emits a chunk when:
    - Sentence boundary is hit (punctuation), OR
    - Word count reaches min_words threshold
    This enables streaming TTS — synthesis starts before the full sentence is done.
    """

    ENDINGS = frozenset({".", "!", "?", ":", ";", "?!", "\n"})

    def __init__(self, min_words: int = 4) -> None:
        self._min_words = min_words
        self._buffer: list[str] = []

    def feed(self, token: str) -> str | None:
        """Feed a token. Returns chunk if ready, else None."""
        self._buffer.append(token)
        combined = "".join(self._buffer)

        # Always emit on sentence boundary
        if any(combined.rstrip().endswith(p) for p in self.ENDINGS):
            self._buffer.clear()
            stripped = combined.strip()
            return stripped if stripped else None

        # Emit when word threshold reached (at a word boundary)
        word_count = len(combined.split())
        if word_count >= self._min_words and token.endswith(" "):
            self._buffer.clear()
            stripped = combined.strip()
            return stripped if stripped else None

        return None

    def flush(self) -> str:
        """Flush remaining buffer."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return text


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
