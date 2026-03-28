"""Remote ASR client — sends audio to faster-whisper-server via HTTP API."""
from __future__ import annotations

import io
import time
import wave
from typing import Any

import httpx
import numpy as np
from loguru import logger
from numpy.typing import NDArray


class RemoteASR:
    """Sends audio to a remote faster-whisper-server for transcription.

    Compatible with OpenAI Whisper API format (POST /v1/audio/transcriptions).
    Falls back to local Whisper if the server is unavailable.
    """

    def __init__(
        self,
        url: str = "http://localhost:8000",
        model: str = "Systran/faster-whisper-large-v3",
        language: str = "ru",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._language = language
        self._timeout = timeout
        self._max_retries = max_retries
        self._available = True
        self._consecutive_failures = 0

    def transcribe(self, audio: NDArray[np.float32], sample_rate: int = 16000) -> str:
        """Transcribe audio array via remote server.

        Args:
            audio: float32 audio samples at given sample_rate.
            sample_rate: Audio sample rate (default 16000 Hz).

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self._available:
            return ""

        # Convert float32 numpy → WAV bytes
        wav_bytes = self._to_wav(audio, sample_rate)

        for attempt in range(1, self._max_retries + 1):
            try:
                t0 = time.perf_counter()
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        f"{self._url}/v1/audio/transcriptions",
                        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                        data={
                            "model": self._model,
                            "language": self._language,
                        },
                    )
                    resp.raise_for_status()

                result = resp.json()
                text = result.get("text", "").strip()
                dt = time.perf_counter() - t0

                self._consecutive_failures = 0
                if text:
                    logger.info("RemoteASR: {:.2f}s → '{}'", dt, text[:80])
                return text

            except httpx.TimeoutException:
                logger.warning(
                    "RemoteASR timeout (attempt {}/{})", attempt, self._max_retries
                )
            except httpx.ConnectError as e:
                logger.warning(
                    "RemoteASR connection failed (attempt {}/{}): {}",
                    attempt, self._max_retries, e,
                )
            except httpx.HTTPStatusError as e:
                logger.error("RemoteASR HTTP error: {} {}", e.response.status_code, e.response.text[:200])
                break  # don't retry on 4xx
            except Exception as e:
                logger.error(
                    "RemoteASR error (attempt {}/{}): {}",
                    attempt, self._max_retries, e,
                )

            if attempt < self._max_retries:
                time.sleep(1.0)

        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            logger.error("RemoteASR: {} failures, disabling", self._consecutive_failures)
            self._available = False
        return ""

    def transcribe_file(self, audio_path: Any) -> str:
        """Transcribe an audio file via remote server."""
        try:
            with open(str(audio_path), "rb") as f:
                wav_bytes = f.read()

            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={
                        "model": self._model,
                        "language": self._language,
                    },
                )
                resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as e:
            logger.error("RemoteASR transcribe_file failed: {}", e)
            return ""

    @staticmethod
    def _to_wav(audio: NDArray[np.float32], sample_rate: int) -> bytes:
        """Convert float32 numpy array to WAV bytes."""
        # Clip and convert to int16
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def check_connection(self) -> bool:
        """Check if the remote ASR server is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._url}/health")
                ok = resp.status_code == 200
                if ok:
                    logger.success("RemoteASR: server healthy at {}", self._url)
                return ok
        except Exception as e:
            logger.warning("RemoteASR: server unreachable: {}", e)
            return False

    @property
    def available(self) -> bool:
        return self._available
