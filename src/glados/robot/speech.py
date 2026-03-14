"""SpeechWorker: Mic → VAD → ASR → LLM queue."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..ASR import TranscriberProtocol
from ..audio_io import AudioProtocol


class SpeechWorker:
    """Listens to microphone, detects speech via VAD, transcribes via ASR,
    and puts results into the LLM priority queue."""

    VAD_SIZE: int = 32
    BUFFER_SIZE: int = 800
    PAUSE_LIMIT: int = 640

    def __init__(
        self,
        audio_io: AudioProtocol,
        asr_model: TranscriberProtocol,
        llm_queue: queue.Queue,
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        listening_event: threading.Event | None = None,
        interruptible: bool = True,
        interrupt_keywords: list[str] | None = None,
    ) -> None:
        self._audio_io = audio_io
        self._asr = asr_model
        self._llm_queue = llm_queue
        self._shutdown = shutdown_event
        self._speaking = currently_speaking_event
        self._listening = listening_event
        self._interruptible = interruptible
        self._interrupt_kw = [kw.lower() for kw in interrupt_keywords] if interrupt_keywords else None

        self._buffer: deque[NDArray[np.float32]] = deque(maxlen=self.BUFFER_SIZE // self.VAD_SIZE)
        self._sample_queue = audio_io.get_sample_queue()
        self._recording = False
        self._samples: list[NDArray[np.float32]] = []
        self._gap_counter = 0
        self._pending_interrupt = False

    def run(self) -> None:
        logger.info("SpeechWorker started.")
        try:
            while not self._shutdown.is_set():
                try:
                    sample, vad_active = self._sample_queue.get(timeout=0.05)
                    self._handle(sample, vad_active)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error("SpeechWorker error: {}", e)
                    self._reset()
        finally:
            logger.info("SpeechWorker stopped.")

    def _handle(self, sample: NDArray[np.float32], vad_active: bool) -> None:
        if vad_active:
            if not self._recording:
                if self._speaking.is_set():
                    if not self._interruptible:
                        return
                    if self._interrupt_kw:
                        self._pending_interrupt = True
                    else:
                        self._audio_io.stop_speaking()
                        self._speaking.clear()
                self._recording = True
                if self._listening:
                    self._listening.set()
                self._samples.extend(self._buffer)
            self._samples.append(sample)
            self._gap_counter = 0
        elif self._recording:
            self._samples.append(sample)
            self._gap_counter += 1
            if self._gap_counter >= self.PAUSE_LIMIT // self.VAD_SIZE:
                self._process()
        else:
            self._buffer.append(sample)
            self._gap_counter = 0

    def _process(self) -> None:
        if not self._samples:
            self._reset()
            return
        audio = np.concatenate(self._samples)
        text = self._asr.transcribe(audio).strip()

        if not text:
            self._reset()
            return

        if self._pending_interrupt and self._interrupt_kw:
            if not any(kw in text.lower() for kw in self._interrupt_kw):
                logger.debug("Ignoring (no keyword): '{}'", text)
                self._reset()
                return
            logger.success("Interrupt keyword in: '{}'", text)
            self._audio_io.stop_speaking()
            self._speaking.clear()

        logger.success("ASR: '{}'", text)
        self._llm_queue.put({
            "role": "user",
            "content": text,
            "_enqueued_at": time.time(),
            "_lane": "priority",
        })
        self._reset()

    def _reset(self) -> None:
        self._recording = False
        if self._listening:
            self._listening.clear()
        self._samples.clear()
        self._gap_counter = 0
        self._pending_interrupt = False
