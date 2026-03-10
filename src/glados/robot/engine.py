"""RobotEngine: wires 5 threads together and manages lifecycle."""
from __future__ import annotations

import queue
import threading

from loguru import logger

from ..ASR import get_audio_transcriber
from ..TTS import get_speech_synthesizer
from ..audio_io import get_audio_system
from ..core.conversation_store import ConversationStore
from ..core.knowledge_store import KnowledgeStore
from ..utils import spoken_text_converter as stc
from ..utils.resources import resource_path
from ..vision.vision_state import VisionState
from .brain import BrainWorker
from .config import RobotConfig
from .speech import SpeechWorker
from .vision import VisionWorker
from .voice import SpeakerWorker, VoiceWorker


class RobotEngine:
    """Minimal 5-thread engine for GLaDOS robot mode."""

    def __init__(
        self,
        config: RobotConfig,
        start_vision: bool = True,
        start_audio: bool = True,
    ) -> None:
        self._config = config
        self._shutdown = threading.Event()
        self._speaking = threading.Event()
        self._processing = threading.Event()

        # Queues
        self._priority_q: queue.Queue = queue.Queue()
        self._autonomy_q: queue.Queue = queue.Queue(maxsize=2)
        self._tts_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()

        # Stores
        self._conv = ConversationStore(
            initial_messages=[{"role": "system", "content": config.personality}]
        )
        self._knowledge: KnowledgeStore | None = None
        if config.knowledge_path:
            self._knowledge = KnowledgeStore(resource_path(config.knowledge_path))

        self._vision_state = VisionState() if start_vision else None

        # Models (lazy init)
        self._asr = None
        self._tts = None
        self._audio_io = None
        self._stc = None

        self._threads: list[threading.Thread] = []
        self._start_vision = start_vision
        self._start_audio = start_audio

    def start(self) -> None:
        """Initialize models and start all worker threads."""
        logger.info("RobotEngine starting...")

        # Init models
        self._tts = get_speech_synthesizer(self._config.voice)
        self._stc = stc.SpokenTextConverter()

        if self._start_audio:
            self._audio_io = get_audio_system("sounddevice")
            self._asr = get_audio_transcriber(
                engine_type=self._config.asr_engine,
            )
            # Warmup ASR
            self._asr.transcribe_file(resource_path("data/0.wav"))
            self._audio_io.start_listening()

        # Vision
        vlm = None
        face_rec = None
        if self._start_vision:
            from ..vision.fastvlm import FastVLM
            vlm = FastVLM(resource_path("models/Vision"))
            try:
                from .face_id import FaceRecognizer
                face_rec = FaceRecognizer(
                    model_dir=resource_path("models/Face"),
                    face_db_dir=self._config.face_db,
                )
            except FileNotFoundError as e:
                logger.warning("FaceID models not found, running without: {}", e)

        # Create workers
        brain = BrainWorker(
            priority_queue=self._priority_q,
            autonomy_queue=self._autonomy_q,
            tts_queue=self._tts_q,
            shutdown_event=self._shutdown,
            speaking_event=self._speaking,
            completion_url=str(self._config.completion_url),
            model_name=self._config.llm_model,
            api_key=self._config.api_key,
            conversation_store=self._conv,
            vision_state=self._vision_state,
            knowledge_store=self._knowledge,
            autonomy_system_prompt=self._config.autonomy.tick_prompt,
            llm_options=self._config.llm_options,
            face_names=self._config.face_names,
        )
        self._processing = brain.processing_event

        voice = VoiceWorker(
            tts_queue=self._tts_q,
            audio_queue=self._audio_q,
            tts_model=self._tts,
            stc=self._stc,
            shutdown_event=self._shutdown,
        )

        thread_targets = [
            ("BrainWorker", brain.run),
            ("VoiceWorker", voice.run),
        ]

        if self._start_audio and self._audio_io and self._asr:
            speech = SpeechWorker(
                audio_io=self._audio_io,
                asr_model=self._asr,
                llm_queue=self._priority_q,
                shutdown_event=self._shutdown,
                currently_speaking_event=self._speaking,
                interruptible=self._config.interruptible,
                interrupt_keywords=self._config.interrupt_keywords,
            )
            speaker = SpeakerWorker(
                audio_io=self._audio_io,
                audio_queue=self._audio_q,
                conversation_store=self._conv,
                sample_rate=self._tts.sample_rate,
                shutdown_event=self._shutdown,
                speaking_event=self._speaking,
                processing_event=self._processing,
            )
            thread_targets.append(("SpeechWorker", speech.run))
            thread_targets.append(("SpeakerWorker", speaker.run))

        if self._start_vision and self._vision_state:
            vision = VisionWorker(
                vision_state=self._vision_state,
                event_queue=self._autonomy_q,
                shutdown_event=self._shutdown,
                settings=self._config.vision,
                face_recognizer=face_rec,
                vlm=vlm,
            )
            thread_targets.append(("VisionWorker", vision.run))

        for name, target in thread_targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            logger.success("{} started.", name)

        # Play announcement
        if self._config.announcement:
            self._tts_q.put(self._config.announcement)
            self._tts_q.put("<EOS>")
            self._processing.set()

        logger.success("RobotEngine ready ({} threads).", len(self._threads))

    def run(self) -> None:
        """Start and block until shutdown."""
        self.start()
        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=0.5)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt.")
        finally:
            self.stop()

    def stop(self) -> None:
        """Signal shutdown and wait for threads."""
        logger.info("Shutting down...")
        self._shutdown.set()
        for t in self._threads:
            t.join(timeout=5.0)
        if self._audio_io:
            self._audio_io.stop_listening()
        logger.info("RobotEngine stopped.")
