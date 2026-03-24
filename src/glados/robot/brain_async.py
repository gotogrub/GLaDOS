"""Async brain: handles speech/vision events, streams LLM, emits TTS events."""
from __future__ import annotations

import threading

from loguru import logger

from ..core.conversation_store import ConversationStore
from ..vision.vision_state import VisionState
from .context import ContextBuilder
from .event_bus import Event, EventBus
from .llm_client import OllamaClient
from .text_pipeline import ChunkSplitter, EmotionParser, ThinkFilter


class AsyncBrain:
    """Processes speech/vision events through LLM and emits TTS sentences."""

    # Keep last N user/assistant message pairs (not counting system messages).
    MAX_CONVERSATION_TURNS: int = 20

    def __init__(
        self,
        event_bus: EventBus,
        llm_client: OllamaClient,
        personality: str,
        context_builder: ContextBuilder | None = None,
        conversation: ConversationStore | None = None,
        vision_state: VisionState | None = None,
        listening_event: threading.Event | None = None,
    ) -> None:
        self._bus = event_bus
        self._llm = llm_client
        self._conv = conversation or ConversationStore(
            initial_messages=[{"role": "system", "content": personality}]
        )
        self._ctx = context_builder or ContextBuilder()
        self._vision = vision_state
        self._listening = listening_event

    async def handle_speech(self, event: Event) -> None:
        """Process user speech through LLM."""
        text = event.data.get("text", "")
        if not text:
            return
        self._conv.append({"role": "user", "content": text})
        await self._generate(autonomy=False)

    async def handle_vision(self, event: Event) -> None:
        """Update vision state only — no autonomous commentary.

        Vision description is stored in VisionState and injected into context
        by ContextBuilder when the user speaks. GLaDOS sees but only comments
        when asked — autonomous vision commentary was too noisy and degraded
        response quality (TTFT growth, repetitive answers).
        """
        # Vision state is already updated by VisionWorker.
        # Nothing to do here — ContextBuilder will include it in next user turn.
        pass

    def _trim_conversation(self) -> None:
        """Keep system messages + last N turns to prevent context growth."""
        msgs = self._conv.snapshot()
        system = [m for m in msgs if m.get("role") == "system"]
        non_system = [m for m in msgs if m.get("role") != "system"]
        max_msgs = self.MAX_CONVERSATION_TURNS * 2  # user + assistant per turn
        if len(non_system) > max_msgs:
            trimmed = system + non_system[-max_msgs:]
            self._conv.replace_all(trimmed)
            logger.info("Trimmed conversation: {} → {} messages", len(msgs), len(trimmed))

    async def _generate(self, autonomy: bool) -> None:
        self._trim_conversation()
        vision_desc = self._vision.snapshot() if self._vision else None
        messages = self._ctx.build(
            messages=self._conv.snapshot(),
            vision_desc=vision_desc,
            autonomy=autonomy,
        )

        think_filter = ThinkFilter()
        emotion_parser = EmotionParser()
        splitter = ChunkSplitter(min_words=4)
        full_response: list[str] = []

        chunks_emitted = 0
        try:
            async for token in self._llm.stream(messages):
                speakable = think_filter.feed(token)
                if not speakable:
                    continue
                # Strip emotion tag, pass clean text downstream
                clean = emotion_parser.feed(speakable)
                if not clean:
                    continue
                full_response.append(clean)
                sentence = splitter.feed(clean)
                if sentence:
                    chunks_emitted += 1
                    logger.info("Brain → TTS chunk {}: '{}'", chunks_emitted, sentence[:80])
                    await self._bus.publish(
                        Event(type="tts", data={"text": sentence}, priority=5)
                    )
        except Exception as e:
            logger.error("AsyncBrain LLM error: {}", e)

        # Flush emotion parser buffer (short responses may not trigger detection)
        leftover = emotion_parser.flush()
        if leftover:
            full_response.append(leftover)
            chunk = splitter.feed(leftover)
            if chunk:
                chunks_emitted += 1
                logger.info("Brain → TTS chunk {}: '{}'", chunks_emitted, chunk[:80])
                await self._bus.publish(
                    Event(type="tts", data={"text": chunk}, priority=5)
                )

        # Publish detected emotion
        await self._bus.publish(
            Event(type="emotion", data={
                "emotion": emotion_parser.emotion,
                "intensity": emotion_parser.intensity,
            }, priority=5)
        )

        # Flush remaining text
        remaining = splitter.flush()
        if remaining:
            chunks_emitted += 1
            logger.info("Brain → TTS flush {}: '{}'", chunks_emitted, remaining[:80])
            await self._bus.publish(
                Event(type="tts", data={"text": remaining}, priority=5)
            )

        # End-of-stream marker
        await self._bus.publish(Event(type="tts_eos", data={}, priority=5))

        full_text = "".join(full_response).strip()
        if full_text:
            self._conv.append({"role": "assistant", "content": full_text})
            logger.success("LLM ({} chunks): {}", chunks_emitted, full_text[:120])
        else:
            logger.warning("LLM returned empty response")
