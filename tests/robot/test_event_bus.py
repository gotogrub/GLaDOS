import asyncio

import pytest
from anyio import sleep

from glados.robot.event_bus import Event, EventBus

pytestmark = pytest.mark.anyio


async def test_publish_subscribe():
    """EventBus should deliver events to subscribers."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe("speech", handler)
    await bus.publish(Event(type="speech", data={"text": "hello"}))
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].data["text"] == "hello"


async def test_no_crosstalk():
    """Subscribers should only receive events of their type."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe("speech", handler)
    await bus.publish(Event(type="vision", data={"desc": "room"}))
    await asyncio.sleep(0.05)

    assert len(received) == 0


async def test_multiple_subscribers():
    """Multiple subscribers for same event type should all receive it."""
    bus = EventBus()
    a, b = [], []

    bus.subscribe("speech", lambda e: a.append(e) or asyncio.sleep(0))
    bus.subscribe("speech", lambda e: b.append(e) or asyncio.sleep(0))
    await bus.publish(Event(type="speech", data={"text": "hi"}))
    await asyncio.sleep(0.05)

    assert len(a) == 1
    assert len(b) == 1
