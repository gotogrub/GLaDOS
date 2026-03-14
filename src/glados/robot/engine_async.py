"""AsyncRobotEngine: asyncio-based engine bridging async brain with threaded I/O."""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from loguru import logger

from ..ASR import get_audio_transcriber
from ..TTS import get_speech_synthesizer
from ..audio_io import get_audio_system
from ..core.conversation_store import ConversationStore
from ..core.knowledge_store import KnowledgeStore
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from ..vision.vision_state import VisionState
from .brain_async import AsyncBrain
from .config import RobotConfig
from .context import ContextBuilder
from .event_bus import Event, EventBus
from .llm_client import OllamaClient
from .speech import SpeechWorker
from .vision import VisionWorker
from .voice import AudioChunk, SpeakerWorker, VoiceWorker
from .face_display import FaceDisplay
from .watchdog import Watchdog


class AsyncRobotEngine:
    """Asyncio-based robot engine.

    Architecture:
    - Main thread: asyncio event loop (EventBus, AsyncBrain, TTS dispatch)
    - Worker threads: SpeechWorker, VisionWorker, SpeakerWorker (blocking I/O)
    - VoiceWorker runs as async task (TTS synthesis via asyncio.to_thread)
    """

    def __init__(
        self,
        config: RobotConfig,
        start_vision: bool = True,
        start_audio: bool = True,
    ) -> None:
        self._config = config
        self._start_vision = start_vision
        self._start_audio = start_audio
        self._shutdown = threading.Event()
        self._speaking = threading.Event()
        self._listening = threading.Event()  # set when ASR is recording speech

        # Async components
        self._bus = EventBus()
        self._conv = ConversationStore(
            initial_messages=[{"role": "system", "content": config.personality}]
        )
        self._vision_state = VisionState() if start_vision else None

        # Thread-safe queues for bridging threads ↔ async
        self._tts_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()

        # Will be initialized in start()
        self._threads: list[threading.Thread] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Initialize models and start workers."""
        logger.info("AsyncRobotEngine starting...")
        self._loop = asyncio.get_running_loop()

        # LLM client
        llm = OllamaClient(
            url=str(self._config.completion_url),
            model=self._config.llm_model,
            api_key=self._config.api_key,
            options=self._config.llm_options,
        )

        # Knowledge
        knowledge: KnowledgeStore | None = None
        if self._config.knowledge_path:
            knowledge = KnowledgeStore(resource_path(self._config.knowledge_path))

        # Context builder
        ctx = ContextBuilder(
            face_profiles=self._config.get_face_profiles(),
            knowledge_store=knowledge,
            autonomy_prompt=self._config.autonomy.tick_prompt,
        )

        # Brain
        brain = AsyncBrain(
            event_bus=self._bus,
            llm_client=llm,
            personality=self._config.personality,
            context_builder=ctx,
            conversation=self._conv,
            vision_state=self._vision_state,
            listening_event=self._listening,
        )

        # Subscribe brain to events
        self._bus.subscribe("speech", brain.handle_speech)
        self._bus.subscribe("vision", brain.handle_vision)

        # Subscribe TTS handler
        self._bus.subscribe("tts", self._on_tts_event)
        self._bus.subscribe("tts_eos", self._on_tts_eos)

        # TTS + audio models
        logger.info("Loading TTS ({})...", self._config.voice)
        tts_model = await asyncio.to_thread(get_speech_synthesizer, self._config.voice)
        logger.info("TTS loaded.")
        stc_converter = stc.SpokenTextConverter()

        # Thread-based workers
        thread_targets: list[tuple[str, Any]] = []

        # Load ASR and warmup BEFORE starting async tasks.
        # CTranslate2 (faster-whisper) holds the GIL during inference,
        # which deadlocks asyncio.to_thread when other tasks are running.
        if self._start_audio:
            audio_io = get_audio_system("sounddevice")
            logger.info("Loading ASR ({})...", self._config.asr_engine)
            asr = get_audio_transcriber(engine_type=self._config.asr_engine)
            logger.info("ASR warmup...")
            asr.transcribe_file(resource_path("data/0.wav"))
            logger.info("ASR warmup done. Starting audio listener...")
            try:
                # start_listening may hang on broken PulseAudio/PipeWire setups.
                # Run in a thread with a timeout so it doesn't block startup.
                listen_thread = threading.Thread(
                    target=audio_io.start_listening, daemon=True
                )
                listen_thread.start()
                listen_thread.join(timeout=5.0)
                if listen_thread.is_alive():
                    logger.warning(
                        "audio_io.start_listening() timed out (5s) — "
                        "microphone may not work. Check PulseAudio/PipeWire."
                    )
                else:
                    logger.info("Audio listener started.")
            except Exception as e:
                logger.warning("Audio listener failed: {}", e)

            # SpeechWorker bridges mic→async via thread-safe callback
            speech = SpeechWorker(
                audio_io=audio_io,
                asr_model=asr,
                llm_queue=self._make_speech_bridge_queue(),
                shutdown_event=self._shutdown,
                currently_speaking_event=self._speaking,
                listening_event=self._listening,
                interruptible=self._config.interruptible,
                interrupt_keywords=self._config.interrupt_keywords,
            )
            thread_targets.append(("SpeechWorker", speech.run))

            # SpeakerWorker (audio playback)
            import sounddevice as sd

            speaker = SpeakerWorker(
                audio_io=audio_io,
                audio_queue=self._audio_q,
                conversation_store=self._conv,
                sample_rate=tts_model.sample_rate,
                shutdown_event=self._shutdown,
                speaking_event=self._speaking,
                processing_event=threading.Event(),
            )
            thread_targets.append(("SpeakerWorker", speaker.run))
            self._audio_io = audio_io
        else:
            self._audio_io = None

        # Face display (unified window — must init before VisionWorker)
        logger.info("Initializing face display...")
        self._face_display: FaceDisplay | None = None
        if self._config.face_display.enabled:
            fd_cfg = self._config.face_display
            self._face_display = FaceDisplay(
                assets_dir=fd_cfg.assets_dir,
                default_emotion=fd_cfg.default_emotion,
                monitor=fd_cfg.monitor,
                width=fd_cfg.width,
                height=fd_cfg.height,
                speaking_event=self._speaking,
            )
            self._bus.subscribe("emotion", self._face_display.handle_emotion_event)
            thread_targets.append((
                "FaceDisplay",
                lambda: self._face_display.run(self._shutdown),
            ))

        # Vision
        if self._start_vision and self._vision_state:
            from ..vision.fastvlm import FastVLM

            logger.info("Loading FastVLM...")
            vlm = FastVLM(resource_path("models/Vision"))
            logger.info("FastVLM loaded.")
            face_rec = None
            try:
                from .face_id import FaceRecognizer

                logger.info("Loading FaceID...")
                face_rec = FaceRecognizer(
                    model_dir=resource_path("models/Face"),
                    face_db_dir=self._config.face_db,
                )
                logger.info("FaceID loaded.")
            except FileNotFoundError as e:
                logger.warning("FaceID models not found: {}", e)

            vision = VisionWorker(
                vision_state=self._vision_state,
                event_queue=self._make_vision_bridge_queue(),
                shutdown_event=self._shutdown,
                settings=self._config.vision,
                face_recognizer=face_rec,
                vlm=vlm,
                face_display=self._face_display,
            )
            thread_targets.append(("VisionWorker", vision.run))

        # Start VoiceLoop AFTER all model loading is done.
        # CTranslate2 and ONNX hold the GIL during init, which would
        # deadlock asyncio.to_thread if async tasks were already running.
        self._voice_task = asyncio.create_task(
            self._voice_loop(tts_model, stc_converter)
        )

        # Start threads
        for name, target in thread_targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.success("{} started.", name)

        # Watchdog
        if self._threads:
            thread_map = {t.name: t for t in self._threads}
            watchdog = Watchdog(
                threads=thread_map,
                on_failure=lambda n, e: logger.critical("CRITICAL: {} died! {}", n, e),
                check_interval=2.0,
                shutdown_event=self._shutdown,
            )
            wd = threading.Thread(target=watchdog.run, name="Watchdog", daemon=True)
            wd.start()

        # Announcement
        if self._config.announcement:
            self._tts_q.put(self._config.announcement)
            self._tts_q.put("<EOS>")

        logger.success(
            "AsyncRobotEngine ready ({} threads + async brain).",
            len(self._threads),
        )

    async def stop(self) -> None:
        """Shutdown all workers and release all resources."""
        logger.info("Shutting down...")
        self._shutdown.set()

        if hasattr(self, "_voice_task"):
            self._voice_task.cancel()
            try:
                await self._voice_task
            except asyncio.CancelledError:
                pass

        for t in self._threads:
            t.join(timeout=3.0)
            if t.is_alive():
                logger.warning("Thread {} did not stop in time.", t.name)

        # Stop all sounddevice streams to release audio device
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        if self._audio_io:
            try:
                self._audio_io.stop_listening()
            except Exception:
                pass

        logger.info("AsyncRobotEngine stopped.")

    def _sync_cleanup(self) -> None:
        """Synchronous cleanup — called when asyncio loop is gone."""
        self._shutdown.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        if self._audio_io:
            try:
                self._audio_io.stop_listening()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    def run(self) -> None:
        """Blocking entry point — runs asyncio event loop."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — cleaning up...")
            self._sync_cleanup()
            logger.info("Cleanup done.")

    async def _run_async(self) -> None:
        await self.start()

        # Handle Ctrl+C gracefully inside asyncio
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            __import__("signal").SIGINT,
            lambda: self._shutdown.set(),
        )

        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(0.5)
        finally:
            await self.stop()

    # --- Bridge helpers ---

    def _make_speech_bridge_queue(self) -> queue.Queue:
        """Create a queue that bridges SpeechWorker (thread) → EventBus (async).

        SpeechWorker puts dicts into this queue. A background task picks them
        up and publishes to the EventBus.
        """
        q: queue.Queue = queue.Queue()
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.create_task(self._speech_bridge(q))
        )
        return q

    async def _speech_bridge(self, q: queue.Queue) -> None:
        """Poll thread-safe queue and forward to async EventBus."""
        while not self._shutdown.is_set():
            try:
                item = await asyncio.to_thread(q.get, timeout=0.1)
            except Exception:
                continue
            text = item.get("content", "")
            if text:
                logger.info("SpeechBridge → EventBus: '{}'", text[:80])
                await self._bus.publish(
                    Event(type="speech", data={"text": text}, priority=10)
                )

    def _make_vision_bridge_queue(self) -> queue.Queue:
        """Create a queue that bridges VisionWorker → EventBus."""
        q: queue.Queue = queue.Queue(maxsize=2)
        asyncio.get_running_loop().call_soon(
            lambda: asyncio.create_task(self._vision_bridge(q))
        )
        return q

    async def _vision_bridge(self, q: queue.Queue) -> None:
        """Poll vision queue and forward to async EventBus."""
        while not self._shutdown.is_set():
            try:
                item = await asyncio.to_thread(q.get, timeout=0.5)
            except Exception:
                continue
            desc = getattr(item, "description", None) or str(item)
            await self._bus.publish(
                Event(type="vision", data={"description": desc}, priority=0)
            )

    async def _on_tts_event(self, event: Event) -> None:
        """Handle TTS event — put text into TTS synthesis queue."""
        text = event.data.get("text", "")
        if text:
            logger.info("TTS queue ← '{}'", text[:80])
            self._tts_q.put(text)

    async def _on_tts_eos(self, event: Event) -> None:
        """Handle TTS end-of-stream."""
        logger.info("TTS queue ← <EOS>")
        self._tts_q.put("<EOS>")

    async def _voice_loop(self, tts_model: Any, stc_converter: Any) -> None:
        """Async TTS synthesis loop — reads from _tts_q, synthesizes, puts into _audio_q."""
        import time

        import numpy as np

        logger.info("VoiceLoop (async) started.")
        while not self._shutdown.is_set():
            try:
                text = await asyncio.to_thread(self._tts_q.get, timeout=0.1)
            except Exception:
                continue

            if text == "<EOS>":
                logger.info("VoiceLoop: EOS → audio queue")
                self._audio_q.put(
                    AudioChunk(
                        audio=np.array([], dtype="float32"),
                        text="",
                        is_eos=True,
                    )
                )
                continue

            if not text.strip():
                continue

            spoken = stc_converter.text_to_spoken(text)
            logger.info("VoiceLoop: synthesizing '{}'", spoken[:60])
            t0 = time.perf_counter()
            try:
                audio = await asyncio.to_thread(tts_model.generate_speech_audio, spoken)
            except Exception as e:
                logger.error("VoiceLoop: TTS synthesis FAILED: {}", e)
                continue
            dt = time.perf_counter() - t0
            logger.info(
                "VoiceLoop: {:.2f}s, {} samples → audio queue",
                dt,
                audio.size if hasattr(audio, "size") else len(audio),
            )

            self._audio_q.put(AudioChunk(audio=audio, text=spoken))

        logger.info("VoiceLoop stopped.")
