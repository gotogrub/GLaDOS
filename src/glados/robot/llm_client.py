"""Async streaming LLM client for Ollama."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx
from loguru import logger


class OllamaClient:
    """Async HTTP client for Ollama streaming chat API."""

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
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama chat API. Yields content strings."""
        data: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if self._options:
            data["options"] = self._options

        t0 = time.perf_counter()
        ttft_logged = False

        async with self._client.stream(
            "POST", self._url, json=data, timeout=self._timeout
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
                    return
                content = chunk.get("message", {}).get("content")
                if content:
                    if not ttft_logged:
                        logger.info("LLM TTFT: {:.2f}s", time.perf_counter() - t0)
                        ttft_logged = True
                    yield content

    async def close(self) -> None:
        await self._client.aclose()
