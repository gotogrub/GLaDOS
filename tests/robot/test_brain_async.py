import pytest
from unittest.mock import AsyncMock, MagicMock

from glados.robot.brain_async import AsyncBrain
from glados.robot.event_bus import Event, EventBus
from glados.robot.llm_client import OllamaClient

pytestmark = pytest.mark.anyio


async def _async_iter(items):
    for item in items:
        yield item


async def test_brain_processes_speech_and_emits_tts():
    """AsyncBrain should process speech and emit TTS events."""
    bus = EventBus()
    tts_events = []

    async def tts_handler(event: Event):
        tts_events.append(event)

    bus.subscribe("tts", tts_handler)
    bus.subscribe("tts_eos", tts_handler)

    mock_llm = MagicMock(spec=OllamaClient)
    mock_llm.stream = MagicMock(return_value=_async_iter(["Привет", "."]))

    brain = AsyncBrain(
        event_bus=bus,
        llm_client=mock_llm,
        personality="Ты ГЛаДОС",
    )

    await brain.handle_speech(Event(type="speech", data={"text": "Привет"}))

    # Should have at least one TTS event + EOS
    tts_texts = [e.data.get("text", "") for e in tts_events if e.type == "tts"]
    eos_count = sum(1 for e in tts_events if e.type == "tts_eos")

    assert len(tts_texts) >= 1
    assert "Привет" in "".join(tts_texts)
    assert eos_count == 1


async def test_brain_filters_think_tags():
    """AsyncBrain should filter <think> blocks from LLM output."""
    bus = EventBus()
    tts_events = []

    async def tts_handler(event: Event):
        tts_events.append(event)

    bus.subscribe("tts", tts_handler)
    bus.subscribe("tts_eos", tts_handler)

    mock_llm = MagicMock(spec=OllamaClient)
    mock_llm.stream = MagicMock(
        return_value=_async_iter(["<think>", "reasoning", "</think>", "Ответ", "."])
    )

    brain = AsyncBrain(
        event_bus=bus,
        llm_client=mock_llm,
        personality="test",
    )

    await brain.handle_speech(Event(type="speech", data={"text": "hi"}))

    tts_texts = [e.data.get("text", "") for e in tts_events if e.type == "tts"]
    combined = "".join(tts_texts)
    assert "reasoning" not in combined
    assert "Ответ" in combined


async def test_brain_empty_speech_ignored():
    """AsyncBrain should ignore empty speech events."""
    bus = EventBus()
    mock_llm = MagicMock(spec=OllamaClient)

    brain = AsyncBrain(
        event_bus=bus,
        llm_client=mock_llm,
        personality="test",
    )

    await brain.handle_speech(Event(type="speech", data={"text": ""}))
    mock_llm.stream.assert_not_called()
