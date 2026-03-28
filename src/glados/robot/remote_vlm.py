"""Remote VLM client — sends images to Ollama vision model via HTTP API."""
from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import httpx
import numpy as np
from loguru import logger
from numpy.typing import NDArray


class RemoteVLM:
    """Sends camera frames to a remote Ollama VLM (e.g. qwen2.5-vl:7b).

    Replaces local FastVLM with a remote GPU-accelerated model that
    supports Russian and provides much better scene descriptions.
    """

    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "qwen2.5vl:7b",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self._url = url
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._available = True
        self._consecutive_failures = 0

    def describe(
        self,
        frame: NDArray[np.uint8],
        prompt: str = "Опиши кратко что ты видишь на изображении. Отвечай на русском.",
        max_tokens: int = 100,
    ) -> str:
        """Send a frame to remote VLM and get description.

        This is a blocking call — intended to run in VisionWorker thread.
        """
        if not self._available:
            return ""

        # Encode frame as JPEG base64
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

        data: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
            },
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(self._url, json=data)
                    resp.raise_for_status()

                result = resp.json()
                content = result.get("message", {}).get("content", "").strip()
                dt = time.perf_counter() - t0

                self._consecutive_failures = 0
                logger.debug("RemoteVLM: {:.1f}s, {} chars", dt, len(content))
                return content

            except httpx.TimeoutException:
                logger.warning(
                    "RemoteVLM timeout (attempt {}/{})", attempt, self._max_retries
                )
            except httpx.ConnectError as e:
                logger.warning(
                    "RemoteVLM connection failed (attempt {}/{}): {}",
                    attempt, self._max_retries, e,
                )
            except Exception as e:
                logger.error(
                    "RemoteVLM error (attempt {}/{}): {}",
                    attempt, self._max_retries, e,
                )

            if attempt < self._max_retries:
                time.sleep(self._retry_delay)

        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            logger.error(
                "RemoteVLM: {} consecutive failures, disabling. "
                "Restart to re-enable.", self._consecutive_failures
            )
            self._available = False
        return ""

    @property
    def available(self) -> bool:
        return self._available

    def check_connection(self) -> bool:
        """Test if the remote VLM server is reachable."""
        try:
            # Use the base URL (strip /api/chat → /api/tags)
            base = self._url.rsplit("/api/", 1)[0]
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                available = any(m["name"] == self._model for m in models)
                if available:
                    logger.success("RemoteVLM: {} available on server", self._model)
                else:
                    names = [m["name"] for m in models]
                    logger.warning(
                        "RemoteVLM: {} not found, available: {}", self._model, names
                    )
                return available
        except Exception as e:
            logger.warning("RemoteVLM: server unreachable: {}", e)
            return False
