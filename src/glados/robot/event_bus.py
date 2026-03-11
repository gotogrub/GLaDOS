"""Async event bus with topic-based pub/sub."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger

EventHandler = Callable[["Event"], Coroutine[Any, Any, None]]


@dataclass
class Event:
    type: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = 0.0


class EventBus:
    """Async event bus with topic-based pub/sub.

    Usage:
        bus = EventBus()
        bus.subscribe("speech", my_handler)
        await bus.publish(Event(type="speech", data={"text": "hi"}))
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        if not event.timestamp:
            event.timestamp = time.time()
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("EventBus handler error for '{}': {}", event.type, e)
