import pytest
from unittest.mock import MagicMock

from glados.robot.config import RobotConfig
from glados.robot.engine_async import AsyncRobotEngine

pytestmark = pytest.mark.anyio


def test_async_engine_constructs():
    """AsyncRobotEngine should construct from config."""
    config = RobotConfig(
        llm_model="test",
        completion_url="http://localhost:11434/api/chat",
    )
    engine = AsyncRobotEngine(config, start_vision=False, start_audio=False)
    assert engine is not None
    assert engine._config.llm_model == "test"


async def test_async_engine_starts_and_stops():
    """AsyncRobotEngine should start and stop cleanly with no audio/vision."""
    config = RobotConfig(
        llm_model="test",
        completion_url="http://localhost:11434/api/chat",
    )
    engine = AsyncRobotEngine(config, start_vision=False, start_audio=False)
    # Should not raise
    await engine.start()
    await engine.stop()
