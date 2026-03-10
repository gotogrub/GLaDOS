"""VoiceWorker (TTS synthesis) + SpeakerWorker (audio playback)."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from loguru import logger
from numpy.typing import NDArray

from ..TTS import SpeechSynthesizerProtocol
from ..audio_io import AudioProtocol
from ..core.conversation_store import ConversationStore
from ..utils import spoken_text_converter as stc


@dataclass
class AudioChunk:
    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False


class VoiceWorker:
    """Reads text from TTS queue, synthesizes audio, puts into audio queue."""

    def __init__(
        self,
        tts_queue: queue.Queue,
        audio_queue: queue.Queue,
        tts_model: SpeechSynthesizerProtocol,
        stc: stc.SpokenTextConverter,
        shutdown_event: threading.Event,
    ) -> None:
        self._tts_q = tts_queue
        self._audio_q = audio_queue
        self._tts = tts_model
        self._stc = stc
        self._shutdown = shutdown_event

    def run(self) -> None:
        logger.info("VoiceWorker started.")
        while not self._shutdown.is_set():
            try:
                text = self._tts_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if text == "<EOS>":
                self._audio_q.put(AudioChunk(
                    audio=np.array([], dtype=np.float32), text="", is_eos=True
                ))
                continue

            if not text.strip():
                continue

            spoken = self._stc.text_to_spoken(text)
            t0 = time.perf_counter()
            audio = self._tts.generate_speech_audio(spoken)
            dt = time.perf_counter() - t0
            logger.debug("TTS: {:.2f}s for '{}'", dt, spoken[:60])
            self._audio_q.put(AudioChunk(audio=audio, text=spoken))

        logger.info("VoiceWorker stopped.")


class SpeakerWorker:
    """Plays audio chunks through the audio system."""

    def __init__(
        self,
        audio_io: AudioProtocol,
        audio_queue: queue.Queue,
        conversation_store: ConversationStore,
        sample_rate: int,
        shutdown_event: threading.Event,
        speaking_event: threading.Event,
        processing_event: threading.Event,
    ) -> None:
        self._audio = audio_io
        self._audio_q = audio_queue
        self._conv = conversation_store
        self._sr = sample_rate
        self._shutdown = shutdown_event
        self._speaking = speaking_event
        self._processing = processing_event

    def run(self) -> None:
        logger.info("SpeakerWorker started.")
        while not self._shutdown.is_set():
            try:
                chunk: AudioChunk = self._audio_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if chunk.is_eos:
                self._speaking.clear()
                self._processing.clear()
                continue

            if chunk.audio.size == 0:
                continue

            self._speaking.set()
            self._audio.start_speaking(chunk.audio, self._sr, chunk.text)

            # Wait for playback to finish (sd.play is non-blocking,
            # _is_playing never resets on its own)
            duration = chunk.audio.size / self._sr
            deadline = time.monotonic() + duration + 0.5
            while time.monotonic() < deadline and not self._shutdown.is_set():
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break
                time.sleep(0.02)
            self._audio.stop_speaking()

        logger.info("SpeakerWorker stopped.")
