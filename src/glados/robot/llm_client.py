"""Async streaming LLM client for Ollama."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx
from loguru import logger


class OllamaClient:
    """Async HTTP client for Ollama streaming chat API.

    Handles two streaming formats:
    1. Standard: all tokens in `content` field (most models)
    2. qwen3: thinking in `thinking` field, answer in `content` field
       In some Ollama versions, qwen3 puts EVERYTHING in `thinking`
       and `content` stays empty. We handle this by yielding from
       whichever field has data.
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._url = url
        self._model = model
        self._options = options or {}
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama chat API.

        Yields all tokens (thinking + content). The downstream ThinkFilter
        strips <think>...</think> blocks. If qwen3 puts everything in the
        `thinking` field, we yield that too — it contains the full response
        including any <think> tags.
        """
        q: asyncio.Queue[str | None] = asyncio.Queue()
        t0 = time.perf_counter()
        ttft_logged = False

        async def _reader() -> None:
            data: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "keep_alive": -1,
                "think": False,  # disable qwen3 thinking (use content only)
            }
            if self._options:
                data["options"] = self._options
            try:
                async with httpx.AsyncClient(
                    headers=self._headers, timeout=self._timeout
                ) as client:
                    async with client.stream(
                        "POST", self._url, json=data
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("done"):
                                break
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                await q.put(content)
            except Exception as e:
                logger.error("OllamaClient stream error: {}", e)
            finally:
                await q.put(None)

        task = asyncio.create_task(_reader())

        try:
            while True:
                token = await q.get()
                if token is None:
                    break
                if not ttft_logged:
                    logger.info(
                        "LLM TTFT: {:.2f}s", time.perf_counter() - t0
                    )
                    ttft_logged = True
                yield token
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def close(self) -> None:
        pass
