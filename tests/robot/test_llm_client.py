import pytest

from glados.robot.llm_client import OllamaClient

pytestmark = pytest.mark.anyio


async def test_stream_tokens(mocker):
    """OllamaClient should yield tokens from streaming response."""
    client = OllamaClient(url="http://localhost:11434/api/chat", model="test")

    mock_lines = [
        '{"message":{"content":"Hello"},"done":false}',
        '{"message":{"content":" world"},"done":false}',
        '{"done":true}',
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    # Mock the httpx stream context
    mock_response = mocker.MagicMock()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_response.aiter_lines = mock_aiter_lines

    mock_ctx = mocker.AsyncMock()
    mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch.object(client._client, "stream", return_value=mock_ctx)

    tokens = []
    async for token in client.stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["Hello", " world"]


async def test_stream_handles_empty_lines(mocker):
    """OllamaClient should skip empty lines."""
    client = OllamaClient(url="http://localhost:11434/api/chat", model="test")

    mock_lines = [
        "",
        '{"message":{"content":"ok"},"done":false}',
        "",
        '{"done":true}',
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    mock_response = mocker.MagicMock()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_response.aiter_lines = mock_aiter_lines

    mock_ctx = mocker.AsyncMock()
    mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch.object(client._client, "stream", return_value=mock_ctx)

    tokens = []
    async for token in client.stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["ok"]
