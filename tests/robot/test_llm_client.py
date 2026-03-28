import pytest

from glados.robot.llm_client import OllamaClient

pytestmark = pytest.mark.anyio


async def test_stream_tokens(mocker):
    """OllamaClient should yield content tokens from streaming response."""
    mock_lines = [
        '{"message":{"content":"Hello"},"done":false}',
        '{"message":{"content":" world"},"done":false}',
        '{"done":true}',
    ]

    async def mock_aiter_lines():
        for line in mock_lines:
            yield line

    mock_response = mocker.MagicMock()
    mock_response.raise_for_status = mocker.MagicMock()
    mock_response.aiter_lines = mock_aiter_lines

    mock_stream_ctx = mocker.AsyncMock()
    mock_stream_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mock_client_instance = mocker.MagicMock()
    mock_client_instance.stream = mocker.MagicMock(return_value=mock_stream_ctx)

    mock_client_ctx = mocker.AsyncMock()
    mock_client_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_client_instance)
    mock_client_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("glados.robot.llm_client.httpx.AsyncClient", return_value=mock_client_ctx)

    client = OllamaClient(url="http://localhost:11434/api/chat", model="test")
    tokens = []
    async for token in client.stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["Hello", " world"]


async def test_stream_skips_empty_lines(mocker):
    """OllamaClient should skip empty lines and thinking-only chunks."""
    mock_lines = [
        "",
        '{"message":{"content":"","thinking":"blah"},"done":false}',
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

    mock_stream_ctx = mocker.AsyncMock()
    mock_stream_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mock_client_instance = mocker.MagicMock()
    mock_client_instance.stream = mocker.MagicMock(return_value=mock_stream_ctx)

    mock_client_ctx = mocker.AsyncMock()
    mock_client_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_client_instance)
    mock_client_ctx.__aexit__ = mocker.AsyncMock(return_value=False)

    mocker.patch("glados.robot.llm_client.httpx.AsyncClient", return_value=mock_client_ctx)

    client = OllamaClient(url="http://localhost:11434/api/chat", model="test")
    tokens = []
    async for token in client.stream([{"role": "user", "content": "hi"}]):
        tokens.append(token)

    assert tokens == ["ok"]
