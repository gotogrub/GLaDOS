"""Async brain: handles speech/vision events, streams LLM, emits TTS events."""
from __future__ import annotations

from loguru import logger

from ..core.conversation_store import ConversationStore
from ..vision.vision_state import VisionState
from .context import ContextBuilder
from .event_bus import Event, EventBus
from .llm_client import OllamaClient
from .text_pipeline import SentenceSplitter, ThinkFilter


class AsyncBrain:
    """Processes speech/vision events through LLM and emits TTS sentences."""

    def __init__(
        self,
        event_bus: EventBus,
        llm_client: OllamaClient,
        personality: str,
        context_builder: ContextBuilder | None = None,
        conversation: ConversationStore | None = None,
        vision_state: VisionState | None = None,
    ) -> None:
        self._bus = event_bus
        self._llm = llm_client
        self._conv = conversation or ConversationStore(
            initial_messages=[{"role": "system", "content": personality}]
        )
        self._ctx = context_builder or ContextBuilder()
        self._vision = vision_state

    async def handle_speech(self, event: Event) -> None:
        """Process user speech through LLM."""
        text = event.data.get("text", "")
        if not text:
            return
        self._conv.append({"role": "user", "content": text})
        await self._generate(autonomy=False)

    async def handle_vision(self, event: Event) -> None:
        """Process vision event (autonomy)."""
        desc = event.data.get("description", "")
        if not desc:
            return
        self._conv.append({"role": "user", "content": desc})
        await self._generate(autonomy=True)

    async def _generate(self, autonomy: bool) -> None:
        vision_desc = self._vision.snapshot() if self._vision else None
        messages = self._ctx.build(
            messages=self._conv.snapshot(),
            vision_desc=vision_desc,
            autonomy=autonomy,
        )

        think_filter = ThinkFilter()
        splitter = SentenceSplitter()
        full_response: list[str] = []

        try:
            async for token in self._llm.stream(messages):
                speakable = think_filter.feed(token)
                if not speakable:
                    continue
                full_response.append(speakable)
                sentence = splitter.feed(speakable)
                if sentence:
                    await self._bus.publish(
                        Event(type="tts", data={"text": sentence}, priority=5)
                    )
        except Exception as e:
            logger.error("AsyncBrain LLM error: {}", e)

        # Flush remaining text
        remaining = splitter.flush()
        if remaining:
            await self._bus.publish(
                Event(type="tts", data={"text": remaining}, priority=5)
            )

        # End-of-stream marker
        await self._bus.publish(Event(type="tts_eos", data={}, priority=5))

        full_text = "".join(full_response).strip()
        if full_text:
            self._conv.append({"role": "assistant", "content": full_text})
            logger.success("LLM: {}", full_text[:120])
