"""Whisper-based ASR with multilingual support (Russian, English, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from loguru import logger


class AudioTranscriber:
    """ASR transcriber using faster-whisper for multilingual speech recognition."""

    DEFAULT_MODEL_SIZE = "base"
    DEFAULT_LANGUAGE = "ru"

    def __init__(
        self,
        model_size: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        **kwargs: Any,
    ) -> None:
        import os

        from faster_whisper import WhisperModel

        # Allow override via env var (e.g. WHISPER_MODEL=medium)
        if model_size is None:
            model_size = os.environ.get("WHISPER_MODEL", self.DEFAULT_MODEL_SIZE)

        self._language = language
        logger.success("Loading Whisper ASR model '{}' (language={})...", model_size, language)
        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
        logger.success("Whisper ASR model loaded.")

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """Transcribe audio array to text.

        Args:
            audio: Audio samples as float32, sample rate 16000 Hz.

        Returns:
            Transcribed text string.
        """
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=3,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.).

        Returns:
            Transcribed text string.
        """
        segments, _ = self._model.transcribe(
            str(audio_path),
            language=self._language,
            beam_size=3,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()
