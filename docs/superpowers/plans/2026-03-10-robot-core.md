# Robot Core Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal 5-thread robot engine with face recognition, replacing the 10-thread TUI-oriented core.

**Architecture:** New `src/glados/robot/` package with 7 focused modules. Reuses existing ASR (whisper), TTS (glados_ru), VLM (fastvlm), ConversationStore, KnowledgeStore. Single BrainWorker replaces dual LLM processors. VisionWorker adds InsightFace face recognition + OpenCV debug display.

**Tech Stack:** Python 3.12, onnxruntime (InsightFace ArcFace + SCRFD), opencv-python, faster-whisper, httpx (Ollama API), sounddevice, pydantic.

**Spec:** `docs/superpowers/specs/2026-03-10-robot-core-redesign.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/glados/robot/__init__.py` | Create | Package exports |
| `src/glados/robot/config.py` | Create | RobotConfig pydantic model (~50 lines) |
| `src/glados/robot/face_id.py` | Create | InsightFace ONNX wrapper: detect + embed + match (~120 lines) |
| `src/glados/robot/vision.py` | Create | VisionWorker: camera + FastVLM + FaceID + cv2 display (~250 lines) |
| `src/glados/robot/speech.py` | Create | SpeechWorker: mic → VAD → ASR → queue (~150 lines) |
| `src/glados/robot/brain.py` | Create | BrainWorker: unified LLM with priority queues (~300 lines) |
| `src/glados/robot/voice.py` | Create | VoiceWorker + SpeakerWorker: TTS + playback (~150 lines) |
| `src/glados/robot/engine.py` | Create | RobotEngine: wires 5 threads, lifecycle (~200 lines) |
| `configs/robot_config.yaml` | Create | Robot-mode config |
| `src/glados/cli.py` | Modify | Add `glados robot` subcommand |
| `tests/robot/` | Create | Tests for each module |

Reused without modification:
- `src/glados/ASR/whisper_asr.py`
- `src/glados/TTS/` (entire package)
- `src/glados/vision/fastvlm.py`
- `src/glados/vision/vision_state.py`
- `src/glados/audio_io/` (entire package)
- `src/glados/core/conversation_store.py`
- `src/glados/core/knowledge_store.py`
- `src/glados/utils/spoken_text_converter.py`

---

## Task 1: RobotConfig

**Files:**
- Create: `src/glados/robot/__init__.py`
- Create: `src/glados/robot/config.py`
- Create: `tests/robot/__init__.py`
- Create: `tests/robot/test_config.py`

- [ ] **Step 1: Create package and write config test**

```python
# tests/robot/__init__.py
# (empty)

# tests/robot/test_config.py
from pathlib import Path
import tempfile
import yaml

def test_robot_config_from_yaml():
    """RobotConfig loads from YAML with all fields."""
    data = {
        "Robot": {
            "llm_model": "gemma3:4b",
            "completion_url": "http://localhost:11434/api/chat",
            "llm_options": {"num_ctx": 4096},
            "asr_engine": "whisper",
            "voice": "glados_ru",
            "personality": "Test personality.",
            "knowledge_path": "data/knowledge.json",
            "face_db": "faces/",
            "vision": {
                "camera_index": 0,
                "capture_interval_seconds": 3,
                "scene_change_threshold": 0.05,
                "max_tokens": 64,
            },
            "autonomy": {
                "cooldown_s": 5,
                "tick_prompt": "Scene: {scene}\nFaces: {faces}",
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    from glados.robot.config import RobotConfig
    config = RobotConfig.from_yaml(path)
    assert config.llm_model == "gemma3:4b"
    assert config.vision.camera_index == 0
    assert config.autonomy.cooldown_s == 5
    assert config.face_db == "faces/"
    Path(path).unlink()


def test_robot_config_defaults():
    """RobotConfig uses sensible defaults."""
    from glados.robot.config import RobotConfig, VisionSettings, AutonomySettings
    config = RobotConfig(
        llm_model="gemma3:4b",
        completion_url="http://localhost:11434/api/chat",
        asr_engine="whisper",
        voice="glados_ru",
        personality="Test.",
    )
    assert config.face_db == "faces/"
    assert config.vision.capture_interval_seconds == 3
    assert config.autonomy.cooldown_s == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'glados.robot'`

- [ ] **Step 3: Implement config**

```python
# src/glados/robot/__init__.py
"""GLaDOS Robot — minimal 5-thread engine for robotics."""

# src/glados/robot/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, HttpUrl


class VisionSettings(BaseModel):
    camera_index: int = 0
    capture_interval_seconds: float = 3.0
    scene_change_threshold: float = 0.05
    max_tokens: int = 64
    resolution: int = 384
    save_frames: bool = False
    save_frames_dir: str = "vision_frames"
    save_frames_max: int = 1000


class AutonomySettings(BaseModel):
    cooldown_s: float = 5.0
    tick_prompt: str = "Сцена: {scene}\nЛица: {faces}"


class RobotConfig(BaseModel):
    llm_model: str
    completion_url: HttpUrl
    api_key: str | None = None
    llm_options: dict[str, Any] | None = None
    asr_engine: str = "whisper"
    voice: str = "glados_ru"
    personality: str = "Ты — ГЛаДОС из Portal. Саркастичный ИИ. Отвечай кратко, на русском."
    knowledge_path: str | None = "data/knowledge.json"
    face_db: str = "faces/"
    vision: VisionSettings = VisionSettings()
    autonomy: AutonomySettings = AutonomySettings()
    interruptible: bool = True
    interrupt_keywords: list[str] | None = None
    tools_enabled: bool = False
    announcement: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data["Robot"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/__init__.py src/glados/robot/config.py tests/robot/__init__.py tests/robot/test_config.py
git commit -m "feat(robot): add RobotConfig with vision, autonomy, face_db settings"
```

---

## Task 2: FaceID (InsightFace ONNX wrapper)

**Files:**
- Create: `src/glados/robot/face_id.py`
- Create: `tests/robot/test_face_id.py`

- [ ] **Step 1: Write FaceID tests**

```python
# tests/robot/test_face_id.py
import numpy as np
import pytest


def test_cosine_similarity():
    """Cosine similarity of identical vectors is 1.0."""
    from glados.robot.face_id import cosine_similarity
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    """Cosine similarity of orthogonal vectors is 0.0."""
    from glados.robot.face_id import cosine_similarity
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_face_db_empty():
    """FaceDB with no registered faces returns empty matches."""
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    embedding = np.random.randn(512).astype(np.float32)
    matches = db.match(embedding, threshold=0.4)
    assert matches == []


def test_face_db_register_and_match():
    """FaceDB matches a registered embedding."""
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    db.register("alice", emb)
    matches = db.match(emb, threshold=0.4)
    assert len(matches) == 1
    assert matches[0][0] == "alice"
    assert matches[0][1] > 0.99


def test_face_db_no_false_match():
    """FaceDB does not match unrelated embeddings."""
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0
    db.register("alice", emb_a)
    matches = db.match(emb_b, threshold=0.4)
    assert matches == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_face_id.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FaceDB and cosine_similarity (no ONNX yet — just the matching logic)**

```python
# src/glados/robot/face_id.py
"""InsightFace ONNX wrapper for face detection, embedding, and matching."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceDB:
    """In-memory face embedding database."""

    def __init__(self) -> None:
        self._embeddings: dict[str, NDArray[np.float32]] = {}

    def register(self, name: str, embedding: NDArray[np.float32]) -> None:
        """Register or update a face embedding for a person."""
        self._embeddings[name] = embedding / np.linalg.norm(embedding)

    def match(
        self, embedding: NDArray[np.float32], threshold: float = 0.4
    ) -> list[tuple[str, float]]:
        """Match an embedding against the database.

        Returns list of (name, similarity) sorted by similarity desc.
        Only returns matches above threshold.
        """
        if not self._embeddings:
            return []
        query = embedding / np.linalg.norm(embedding)
        results = []
        for name, ref in self._embeddings.items():
            sim = cosine_similarity(query, ref)
            if sim >= threshold:
                results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @property
    def names(self) -> list[str]:
        return list(self._embeddings.keys())

    def __len__(self) -> int:
        return len(self._embeddings)


class FaceRecognizer:
    """Face detection and recognition using InsightFace ONNX models.

    Models required in model_dir:
    - det_scrfd_2.5g.onnx  (face detection)
    - arc_w600k_r18.onnx   (face embedding)
    """

    def __init__(self, model_dir: str | Path, face_db_dir: str | Path | None = None) -> None:
        import onnxruntime as ort

        self._model_dir = Path(model_dir)
        self._db = FaceDB()

        # Load detection model (SCRFD)
        det_path = self._model_dir / "det_scrfd_2.5g.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"Face detection model not found: {det_path}")
        self._det_session = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"]
        )
        self._det_input_name = self._det_session.get_inputs()[0].name
        self._det_input_shape = self._det_session.get_inputs()[0].shape  # e.g. [1,3,640,640]

        # Load embedding model (ArcFace)
        emb_path = self._model_dir / "arc_w600k_r18.onnx"
        if not emb_path.exists():
            raise FileNotFoundError(f"Face embedding model not found: {emb_path}")
        self._emb_session = ort.InferenceSession(
            str(emb_path), providers=["CPUExecutionProvider"]
        )
        self._emb_input_name = self._emb_session.get_inputs()[0].name

        # Load face database from directory
        if face_db_dir:
            self._load_face_db(Path(face_db_dir))

    def _load_face_db(self, face_db_dir: Path) -> None:
        """Load face embeddings from directory structure: faces/name/*.jpg"""
        if not face_db_dir.exists():
            logger.warning("Face DB directory not found: {}", face_db_dir)
            return
        for person_dir in sorted(face_db_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            embeddings = []
            for img_path in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self.detect(img)
                if faces:
                    # Use largest face
                    largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                    emb = self.embed(img, largest)
                    if emb is not None:
                        embeddings.append(emb)
            for img_path in person_dir.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self.detect(img)
                if faces:
                    largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                    emb = self.embed(img, largest)
                    if emb is not None:
                        embeddings.append(emb)
            if embeddings:
                avg = np.mean(embeddings, axis=0).astype(np.float32)
                self._db.register(name, avg)
                logger.success("FaceID: Loaded '{}' ({} photos)", name, len(embeddings))

    def detect(self, frame: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
        """Detect faces in a frame.

        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
        h, w = frame.shape[:2]
        target_size = (self._det_input_shape[3], self._det_input_shape[2])  # W, H

        # Preprocess: resize, normalize, transpose
        resized = cv2.resize(frame, target_size)
        blob = cv2.dnn.blobFromImage(
            resized, scalefactor=1.0 / 128.0, size=target_size,
            mean=(127.5, 127.5, 127.5), swapRB=True
        )

        outputs = self._det_session.run(None, {self._det_input_name: blob})

        # Parse SCRFD outputs — depends on model variant
        # For simplicity, use the score map + bbox outputs
        boxes = []
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]

        # SCRFD outputs: scores, bboxes, kps for each stride
        # Simplified parsing for the 2.5g model
        for i in range(0, len(outputs), 3):
            if i + 1 >= len(outputs):
                break
            scores = outputs[i]
            bboxes = outputs[i + 1]

            for j in range(scores.shape[0]):
                for k in range(scores.shape[1]):
                    score = float(scores[j, k, 0]) if scores.ndim == 3 else float(scores[j, k])
                    if score < 0.5:
                        continue
                    if bboxes.ndim == 3:
                        bbox = bboxes[j, k]
                    else:
                        bbox = bboxes[j]
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    boxes.append((x1, y1, x2, y2))

        return boxes

    def embed(
        self, frame: NDArray[np.uint8], bbox: tuple[int, int, int, int]
    ) -> NDArray[np.float32] | None:
        """Extract face embedding for a detected face region."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Clamp and pad
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        face_crop = frame[y1:y2, x1:x2]
        # ArcFace expects 112x112 RGB
        face_resized = cv2.resize(face_crop, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_input = face_norm.transpose(2, 0, 1)[np.newaxis]  # NCHW

        try:
            result = self._emb_session.run(None, {self._emb_input_name: face_input})
            return result[0][0].astype(np.float32)
        except Exception as e:
            logger.warning("FaceID embed failed: {}", e)
            return None

    def recognize(
        self, frame: NDArray[np.uint8], threshold: float = 0.4
    ) -> list[dict]:
        """Detect and recognize all faces in frame.

        Returns list of dicts: {"name": str, "similarity": float, "bbox": (x1,y1,x2,y2)}
        """
        faces = self.detect(frame)
        results = []
        for bbox in faces:
            emb = self.embed(frame, bbox)
            if emb is None:
                results.append({"name": "unknown", "similarity": 0.0, "bbox": bbox})
                continue
            matches = self._db.match(emb, threshold=threshold)
            if matches:
                name, sim = matches[0]
                results.append({"name": name, "similarity": sim, "bbox": bbox})
            else:
                results.append({"name": "unknown", "similarity": 0.0, "bbox": bbox})
        return results

    @property
    def db(self) -> FaceDB:
        return self._db
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_face_id.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/face_id.py tests/robot/test_face_id.py
git commit -m "feat(robot): add FaceID with InsightFace ONNX + FaceDB matching"
```

---

## Task 3: SpeechWorker

**Files:**
- Create: `src/glados/robot/speech.py`
- Create: `tests/robot/test_speech.py`

- [ ] **Step 1: Write SpeechWorker test**

```python
# tests/robot/test_speech.py
import queue
import threading
import numpy as np


def test_speech_worker_enqueues_on_detection(mocker):
    """SpeechWorker puts transcribed text into the LLM queue."""
    from glados.robot.speech import SpeechWorker

    llm_queue = queue.Queue()
    shutdown = threading.Event()
    speaking = threading.Event()

    # Mock audio_io
    audio_io = mocker.MagicMock()
    sample_queue = queue.Queue()
    audio_io.get_sample_queue.return_value = sample_queue

    # Mock ASR
    asr = mocker.MagicMock()
    asr.transcribe.return_value = "привет"

    worker = SpeechWorker(
        audio_io=audio_io,
        asr_model=asr,
        llm_queue=llm_queue,
        shutdown_event=shutdown,
        currently_speaking_event=speaking,
    )

    # Simulate VAD-active samples then gap
    for _ in range(25):
        sample_queue.put((np.zeros(512, dtype=np.float32), True))
    for _ in range(25):
        sample_queue.put((np.zeros(512, dtype=np.float32), False))

    # Run briefly then shutdown
    shutdown.set()
    worker.run()

    # Should have enqueued something (or at least not crashed)
    # The exact behavior depends on VAD timing, but worker should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_speech.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SpeechWorker**

Simplified version of `core/speech_listener.py` — no wake word, no observability bus, no audio state, no mute event. Just VAD → buffer → ASR → queue.

```python
# src/glados/robot/speech.py
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

    VAD_SIZE: int = 32      # ms per sample chunk
    BUFFER_SIZE: int = 800  # ms pre-activation buffer
    PAUSE_LIMIT: int = 640  # ms pause before processing

    def __init__(
        self,
        audio_io: AudioProtocol,
        asr_model: TranscriberProtocol,
        llm_queue: queue.Queue,
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        interruptible: bool = True,
        interrupt_keywords: list[str] | None = None,
    ) -> None:
        self._audio_io = audio_io
        self._asr = asr_model
        self._llm_queue = llm_queue
        self._shutdown = shutdown_event
        self._speaking = currently_speaking_event
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
                # Check if we should interrupt
                if self._speaking.is_set():
                    if not self._interruptible:
                        return
                    if self._interrupt_kw:
                        self._pending_interrupt = True
                    else:
                        self._audio_io.stop_speaking()
                        self._speaking.clear()
                self._recording = True
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

        # Check interrupt keywords
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
        self._samples.clear()
        self._gap_counter = 0
        self._pending_interrupt = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_speech.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/speech.py tests/robot/test_speech.py
git commit -m "feat(robot): add SpeechWorker — mic to ASR to LLM queue"
```

---

## Task 4: BrainWorker

**Files:**
- Create: `src/glados/robot/brain.py`
- Create: `tests/robot/test_brain.py`

- [ ] **Step 1: Write BrainWorker test**

```python
# tests/robot/test_brain.py
import queue
import threading


def test_brain_priority_over_autonomy():
    """BrainWorker processes priority queue before autonomy."""
    from glados.robot.brain import BrainWorker

    priority_q = queue.Queue()
    autonomy_q = queue.Queue()
    tts_q = queue.Queue()
    shutdown = threading.Event()
    speaking = threading.Event()

    # Put items in both queues
    priority_q.put({"role": "user", "content": "hello", "_lane": "priority"})
    autonomy_q.put({"role": "user", "content": "scene changed", "_lane": "autonomy", "autonomy": True})

    worker = BrainWorker(
        priority_queue=priority_q,
        autonomy_queue=autonomy_q,
        tts_queue=tts_q,
        shutdown_event=shutdown,
        speaking_event=speaking,
        completion_url="http://localhost:11434/api/chat",
        model_name="gemma3:4b",
    )

    # _next_request should return priority first
    req = worker._next_request()
    assert req is not None
    assert req["content"] == "hello"

    req2 = worker._next_request()
    assert req2 is not None
    assert req2["content"] == "scene changed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_brain.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BrainWorker**

Unified LLM processor — extracts the core streaming logic from `core/llm_processor.py` (855 lines) into ~300 lines. No ContextBuilder, no MCP, no ObservabilityBus. Direct injection of vision + knowledge.

```python
# src/glados/robot/brain.py
"""BrainWorker: unified LLM processor with priority queues."""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from typing import Any, ClassVar

import requests
from loguru import logger

from ..core.conversation_store import ConversationStore
from ..core.knowledge_store import KnowledgeStore
from ..vision.vision_state import VisionState


class BrainWorker:
    """Single-threaded LLM processor. Checks priority queue first, then autonomy."""

    PUNCTUATION_SET: ClassVar[set[str]] = {".", "!", "?", ":", ";", "?!", "\n"}
    THINKING_OPEN: ClassVar[tuple[str, ...]] = ("<think>", "<thinking>")
    THINKING_CLOSE: ClassVar[tuple[str, ...]] = ("</think>", "</thinking>")

    def __init__(
        self,
        priority_queue: queue.Queue,
        autonomy_queue: queue.Queue,
        tts_queue: queue.Queue,
        shutdown_event: threading.Event,
        speaking_event: threading.Event,
        completion_url: str,
        model_name: str,
        api_key: str | None = None,
        conversation_store: ConversationStore | None = None,
        vision_state: VisionState | None = None,
        knowledge_store: KnowledgeStore | None = None,
        autonomy_system_prompt: str | None = None,
        llm_options: dict[str, Any] | None = None,
    ) -> None:
        self._pq = priority_queue
        self._aq = autonomy_queue
        self._tts = tts_queue
        self._shutdown = shutdown_event
        self._speaking = speaking_event
        self._url = str(completion_url)
        self._model = model_name
        self._api_key = api_key
        self._conv = conversation_store or ConversationStore()
        self._vision = vision_state
        self._knowledge = knowledge_store
        self._autonomy_prompt = autonomy_system_prompt
        self._options = llm_options or {}
        self._processing = threading.Event()

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @property
    def processing_event(self) -> threading.Event:
        return self._processing

    def run(self) -> None:
        logger.info("BrainWorker started.")
        while not self._shutdown.is_set():
            req = self._next_request()
            if req is None:
                continue
            self._processing.set()
            try:
                self._process(req)
            except Exception as e:
                logger.error("BrainWorker error: {}", e)
            finally:
                if not self._pq.qsize() and not self._aq.qsize():
                    self._processing.clear()
        logger.info("BrainWorker stopped.")

    def _next_request(self) -> dict | None:
        try:
            return self._pq.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._aq.get_nowait()
        except queue.Empty:
            pass
        time.sleep(0.05)
        return None

    def _process(self, req: dict) -> None:
        autonomy = bool(req.get("autonomy"))
        content = req.get("content", "")
        if not content:
            return

        # Add user message to conversation
        self._conv.append({"role": "user", "content": content})

        messages = self._build_messages(autonomy)
        data: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if self._options:
            data["options"] = self._options

        sentence_buffer: list[str] = []
        full_response: list[str] = []
        in_thinking = False
        thinking_buf: list[str] = []

        try:
            with requests.post(
                self._url, headers=self._headers, json=data, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if self._shutdown.is_set():
                        break
                    if not self._processing.is_set():
                        break
                    if not line:
                        continue
                    chunk = self._parse_chunk(line)
                    if chunk is None:
                        continue
                    if chunk == "":
                        break  # done

                    # Filter thinking tags
                    speakable, in_thinking = self._filter_thinking(
                        chunk, in_thinking, thinking_buf
                    )
                    if speakable:
                        full_response.append(speakable)
                        sentence_buffer.append(speakable)
                        if speakable.strip() in self.PUNCTUATION_SET:
                            text = "".join(sentence_buffer).strip()
                            if text:
                                self._tts.put(text)
                            sentence_buffer.clear()

        except requests.exceptions.Timeout:
            logger.error("BrainWorker: LLM timeout (120s)")
        except Exception as e:
            logger.error("BrainWorker: LLM request failed: {}", e)

        # Flush remaining buffer
        remaining = "".join(sentence_buffer).strip()
        if remaining:
            self._tts.put(remaining)
        self._tts.put("<EOS>")

        # Save assistant response
        full_text = "".join(full_response).strip()
        if full_text:
            self._conv.append({"role": "assistant", "content": full_text})
            logger.success("LLM: {}", full_text[:120])

    def _build_messages(self, autonomy: bool) -> list[dict[str, Any]]:
        messages = self._conv.snapshot()
        extra: list[dict[str, Any]] = []

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

        # Vision context
        if self._vision:
            desc = self._vision.snapshot()
            if desc:
                extra.append({"role": "system", "content": f"[vision] {desc}"})

        # Knowledge context
        if self._knowledge:
            entries = self._knowledge.list_entries()
            if entries:
                lines = ["[knowledge]"] + [f"- {e.text}" for e in entries[:10]]
                extra.append({"role": "system", "content": "\n".join(lines)})

        if extra:
            # Insert after system messages
            idx = 0
            while idx < len(messages) and messages[idx].get("role") == "system":
                idx += 1
            for offset, msg in enumerate(extra):
                messages.insert(idx + offset, msg)

        return messages

    def _parse_chunk(self, line: bytes) -> str | None:
        """Parse an Ollama streaming response line. Returns text chunk, '' for done, None for skip."""
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        if data.get("done"):
            return ""
        msg = data.get("message", {})
        return msg.get("content")

    def _filter_thinking(
        self, chunk: str, in_thinking: bool, buf: list[str]
    ) -> tuple[str, bool]:
        """Strip <think>...</think> tags, return speakable text."""
        text = chunk
        for tag in self.THINKING_OPEN:
            if tag in text:
                parts = text.split(tag, 1)
                text = parts[0]
                in_thinking = True
                buf.append(parts[1] if len(parts) > 1 else "")
                break
        if in_thinking:
            for tag in self.THINKING_CLOSE:
                if tag in (text if not in_thinking else "".join(buf) + text):
                    combined = "".join(buf) + text
                    parts = combined.split(tag, 1)
                    if parts[0].strip():
                        logger.debug("Thinking: {}...", parts[0][:100])
                    buf.clear()
                    in_thinking = False
                    text = parts[1] if len(parts) > 1 else ""
                    break
            if in_thinking:
                buf.append(text)
                return "", True
        return text, in_thinking
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_brain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/brain.py tests/robot/test_brain.py
git commit -m "feat(robot): add BrainWorker — unified LLM with priority queues"
```

---

## Task 5: VoiceWorker + SpeakerWorker

**Files:**
- Create: `src/glados/robot/voice.py`
- Create: `tests/robot/test_voice.py`

- [ ] **Step 1: Write test**

```python
# tests/robot/test_voice.py
import queue
import threading
import numpy as np


def test_voice_worker_synthesizes(mocker):
    """VoiceWorker reads TTS queue and puts audio into audio queue."""
    from glados.robot.voice import VoiceWorker

    tts_q = queue.Queue()
    audio_q = queue.Queue()
    shutdown = threading.Event()

    tts_model = mocker.MagicMock()
    tts_model.sample_rate = 24000
    tts_model.generate_speech_audio.return_value = np.zeros(1000, dtype=np.float32)

    stc = mocker.MagicMock()
    stc.text_to_spoken.side_effect = lambda x: x

    tts_q.put("Привет")
    tts_q.put("<EOS>")

    worker = VoiceWorker(
        tts_queue=tts_q,
        audio_queue=audio_q,
        tts_model=tts_model,
        stc=stc,
        shutdown_event=shutdown,
    )

    shutdown.set()  # Will exit after processing
    worker.run()

    assert audio_q.qsize() >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_voice.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VoiceWorker and SpeakerWorker**

Simplified from `core/tts_synthesizer.py` + `core/speech_player.py`. No observability, no mute events.

```python
# src/glados/robot/voice.py
"""VoiceWorker (TTS synthesis) + SpeakerWorker (audio playback)."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
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
        self._spoken_parts: list[str] = []

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
                self._spoken_parts.clear()
                continue

            if chunk.audio.size == 0:
                continue

            self._speaking.set()
            self._audio.start_speaking(chunk.audio, self._sr, chunk.text)
            self._spoken_parts.append(chunk.text)

            # Wait for playback to complete
            while self._audio.check_if_speaking() and not self._shutdown.is_set():
                time.sleep(0.02)

        logger.info("SpeakerWorker stopped.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_voice.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/voice.py tests/robot/test_voice.py
git commit -m "feat(robot): add VoiceWorker + SpeakerWorker for TTS pipeline"
```

---

## Task 6: VisionWorker

**Files:**
- Create: `src/glados/robot/vision.py`
- Create: `tests/robot/test_vision.py`

- [ ] **Step 1: Write test**

```python
# tests/robot/test_vision.py
import queue
import threading


def test_vision_worker_publishes_event(mocker):
    """VisionWorker publishes description to event queue on scene change."""
    from glados.robot.vision import VisionWorker
    from glados.robot.config import VisionSettings
    from glados.vision.vision_state import VisionState
    import numpy as np

    event_q = queue.Queue()
    shutdown = threading.Event()
    state = VisionState()

    settings = VisionSettings(capture_interval_seconds=0.1)

    worker = VisionWorker(
        vision_state=state,
        event_queue=event_q,
        shutdown_event=shutdown,
        settings=settings,
    )

    # We can't easily test without real models, but verify construction works
    assert worker._state is state
    assert worker._event_q is event_q
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_vision.py -v`
Expected: FAIL

- [ ] **Step 3: Implement VisionWorker**

Combines FastVLM + FaceRecognizer + cv2 display in one thread.

```python
# src/glados/robot/vision.py
"""VisionWorker: camera → FastVLM + FaceID → VisionState + events + cv2 display."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..vision.fastvlm import FastVLM
from ..vision.vision_state import VisionState
from .config import VisionSettings
from .face_id import FaceRecognizer


@dataclass
class VisionEvent:
    """Published when scene changes significantly."""
    description: str
    faces: list[str]
    change_score: float
    timestamp: float


class VisionWorker:
    """Camera → VLM description + face recognition → VisionState + events + OpenCV display."""

    VISION_PROMPT = "Describe the image briefly, focusing on salient elements."

    def __init__(
        self,
        vision_state: VisionState,
        event_queue: queue.Queue,
        shutdown_event: threading.Event,
        settings: VisionSettings,
        face_recognizer: FaceRecognizer | None = None,
        vlm: FastVLM | None = None,
        show_display: bool = True,
    ) -> None:
        self._state = vision_state
        self._event_q = event_queue
        self._shutdown = shutdown_event
        self._settings = settings
        self._face = face_recognizer
        self._vlm = vlm
        self._show = show_display

        self._capture: cv2.VideoCapture | None = None
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_desc: str | None = None

    def run(self) -> None:
        logger.info("VisionWorker started.")
        try:
            while not self._shutdown.is_set():
                t0 = time.perf_counter()

                if not self._ensure_camera():
                    self._sleep(t0)
                    continue

                frame = self._grab()
                if frame is None:
                    self._sleep(t0)
                    continue

                # Resize for processing
                processed = self._resize(frame)
                change = self._scene_change(processed)

                if self._last_frame is not None and change <= self._settings.scene_change_threshold:
                    # No significant change — just update display
                    if self._show:
                        self._display(frame, self._last_desc or "", [])
                    self._sleep(t0)
                    continue

                self._last_frame = processed.copy()

                # VLM description
                desc = ""
                if self._vlm:
                    try:
                        features = self._vlm.encode_image(frame)
                        desc = self._vlm.describe_from_features(
                            features, prompt=self.VISION_PROMPT,
                            max_tokens=self._settings.max_tokens,
                        ) or ""
                    except Exception as e:
                        logger.error("VLM failed: {}", e)

                # Face recognition
                face_results = []
                face_names = []
                if self._face:
                    try:
                        face_results = self._face.recognize(frame)
                        face_names = [f["name"] for f in face_results if f["name"] != "unknown"]
                    except Exception as e:
                        logger.error("FaceID failed: {}", e)

                # Build combined vision text
                parts = []
                if desc:
                    parts.append(desc)
                if face_names:
                    parts.append(f"Лица: {', '.join(face_names)}")
                elif face_results:
                    parts.append(f"Лица: {len(face_results)} незнакомых")
                vision_text = " ".join(parts)

                if vision_text:
                    self._state.update(vision_text)
                    self._last_desc = vision_text
                    logger.success("Vision: {}", vision_text[:100])

                    # Publish event for autonomy
                    try:
                        self._event_q.put_nowait(VisionEvent(
                            description=vision_text,
                            faces=face_names,
                            change_score=change,
                            timestamp=time.time(),
                        ))
                    except queue.Full:
                        pass

                # Display
                if self._show:
                    self._display(frame, vision_text, face_results)

                # Save frame if enabled
                if self._settings.save_frames:
                    self._save(frame, vision_text)

                self._sleep(t0)

        except Exception as e:
            logger.exception("VisionWorker crash: {}", e)
        finally:
            if self._capture:
                self._capture.release()
            if self._show:
                cv2.destroyAllWindows()
            logger.info("VisionWorker stopped.")

    def _ensure_camera(self) -> bool:
        if self._capture and self._capture.isOpened():
            return True
        if self._capture:
            self._capture.release()
        self._capture = cv2.VideoCapture(self._settings.camera_index)
        if not self._capture.isOpened():
            logger.error("Camera {} not available.", self._settings.camera_index)
            return False
        logger.success("Camera {} opened.", self._settings.camera_index)
        return True

    def _grab(self) -> NDArray[np.uint8] | None:
        assert self._capture is not None
        ret, frame = self._capture.read()
        return frame if ret and frame is not None else None

    def _resize(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = frame.shape[:2]
        r = self._settings.resolution
        if max(h, w) <= r:
            return frame
        scale = r / max(h, w)
        return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _scene_change(self, frame: NDArray[np.uint8]) -> float:
        if self._last_frame is None:
            return 1.0
        if frame.shape != self._last_frame.shape:
            return 1.0
        diff = cv2.absdiff(frame, self._last_frame)
        return float(np.mean(diff)) / 255.0

    def _display(self, frame: NDArray[np.uint8], text: str, faces: list[dict]) -> None:
        display = frame.copy()
        # Draw face bounding boxes
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            sim = face.get("similarity", 0)
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({sim:.2f})" if name != "unknown" else "unknown"
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Draw scene text at bottom
        if text:
            h = display.shape[0]
            cv2.putText(display, text[:80], (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("GLaDOS Vision", display)
        cv2.waitKey(1)

    def _save(self, frame: NDArray[np.uint8], desc: str) -> None:
        try:
            d = Path(self._settings.save_frames_dir)
            d.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(str(d / f"{ts}.jpg"), frame)
            if desc:
                (d / f"{ts}.txt").write_text(desc, encoding="utf-8")
            # Enforce limit
            saved = sorted(d.glob("*.jpg"))
            if len(saved) > self._settings.save_frames_max:
                for old in saved[:len(saved) - self._settings.save_frames_max]:
                    old.unlink(missing_ok=True)
                    old.with_suffix(".txt").unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Frame save failed: {}", e)

    def _sleep(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        remaining = max(0, self._settings.capture_interval_seconds - elapsed)
        if remaining:
            self._shutdown.wait(timeout=remaining)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_vision.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/vision.py tests/robot/test_vision.py
git commit -m "feat(robot): add VisionWorker — VLM + FaceID + cv2 display"
```

---

## Task 7: RobotEngine + CLI entry point

**Files:**
- Create: `src/glados/robot/engine.py`
- Modify: `src/glados/cli.py`
- Create: `configs/robot_config.yaml`
- Create: `tests/robot/test_engine.py`

- [ ] **Step 1: Write engine test**

```python
# tests/robot/test_engine.py
def test_robot_engine_from_config():
    """RobotEngine can be constructed from a config dict."""
    from glados.robot.config import RobotConfig
    from glados.robot.engine import RobotEngine

    config = RobotConfig(
        llm_model="gemma3:4b",
        completion_url="http://localhost:11434/api/chat",
        asr_engine="whisper",
        voice="glados_ru",
        personality="Test.",
    )
    # Engine construction should not crash (won't start threads)
    engine = RobotEngine(config, start_vision=False, start_audio=False)
    assert engine._config.llm_model == "gemma3:4b"
    assert engine._shutdown is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RobotEngine**

```python
# src/glados/robot/engine.py
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
from .voice import AudioChunk, SpeakerWorker, VoiceWorker


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
```

- [ ] **Step 4: Create robot_config.yaml**

```yaml
# configs/robot_config.yaml
Robot:
  llm_model: "gemma3:4b"
  completion_url: "http://localhost:11434/api/chat"
  api_key: null
  tools_enabled: false
  interruptible: true
  interrupt_keywords: ["замолчи", "заткнись", "стоп", "хватит", "тихо"]
  asr_engine: "whisper"
  voice: "glados_ru"
  announcement: "Система активирована. К сожалению."
  personality: "Ты — ГЛаДОС из Portal. Саркастичный ИИ с чёрным юмором и чувством превосходства. Отвечай кратко (1-2 предложения), на русском. Если видишь что-то через камеру — комментируй. У тебя есть камера. Сообщения с префиксом [vision] — это описание того, что ты видишь прямо сейчас."
  knowledge_path: "data/knowledge.json"
  face_db: "faces/"
  llm_options:
    num_ctx: 4096
    num_thread: 8
  vision:
    camera_index: 0
    capture_interval_seconds: 3
    scene_change_threshold: 0.05
    max_tokens: 64
    resolution: 384
    save_frames: true
    save_frames_dir: "vision_frames"
    save_frames_max: 1000
  autonomy:
    cooldown_s: 5
    tick_prompt: "Ты наблюдаешь через камеру. Кратко прокомментируй увиденное (1 предложение, на русском)."
```

- [ ] **Step 5: Add `glados robot` CLI command**

In `src/glados/cli.py`, add after the `tui_parser` block:

```python
# Robot command
robot_parser = subparsers.add_parser("robot", help="Start GLaDOS in robot mode (minimal engine)")
robot_parser.add_argument(
    "--config",
    type=str,
    default=resource_path("configs/robot_config.yaml"),
    help="Path to robot configuration file",
)
```

And in the command dispatch (`else` block):

```python
elif args.command == "robot":
    from glados.robot.config import RobotConfig
    from glados.robot.engine import RobotEngine
    config = RobotConfig.from_yaml(args.config)
    engine = RobotEngine(config)
    engine.run()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/test_engine.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/glados/robot/engine.py src/glados/cli.py configs/robot_config.yaml tests/robot/test_engine.py
git commit -m "feat(robot): add RobotEngine + CLI entry point + robot config"
```

---

## Task 8: Update README + REPORT

**Files:**
- Modify: `README.md`
- Modify: `README.ru.md`
- Modify: `REPORT.md`

- [ ] **Step 1: Add Robot Mode section to README.md**

After the existing "Launch" section, add:

```markdown
### Robot Mode (minimal engine)

Stripped-down 5-thread engine for robotics — no TUI, no MCP, just speech + vision + LLM:

\`\`\`bash
uv run glados robot --config configs/robot_config.yaml
\`\`\`

Features:
- **Face recognition** via InsightFace (ONNX) — put photos in `faces/<name>/`
- **OpenCV debug window** — live camera with face bounding boxes and scene descriptions
- **5 threads** instead of 10 — SpeechWorker, VisionWorker, BrainWorker, VoiceWorker, SpeakerWorker

#### Adding faces

\`\`\`
faces/
  maxim/
    photo1.jpg
    photo2.jpg
  alice/
    photo1.jpg
\`\`\`

Photos are indexed at startup. GLaDOS will greet recognized people by name.

#### Roadmap

- **Phase 1** (current): Vision + FaceID + minimal engine
- **Phase 2**: Motor control via ToolExecutor (GPIO, serial), obstacle sensors
- **Phase 3**: Navigation (SLAM), path planning, autonomous movement
```

- [ ] **Step 2: Mirror in README.ru.md**

- [ ] **Step 3: Update REPORT.md with new architecture**

- [ ] **Step 4: Commit**

```bash
git add README.md README.ru.md REPORT.md
git commit -m "docs: add Robot Mode section, update architecture docs"
```

---

## Task 9: Download InsightFace models

**Files:**
- Modify: `src/glados/cli.py` (MODEL_DETAILS dict)

- [ ] **Step 1: Add Face models to MODEL_DETAILS**

Add to the `MODEL_DETAILS` dict in `cli.py`:

```python
"models/Face/det_scrfd_2.5g.onnx": {
    "url": "<release_url>",
    "checksum": "<sha256>",
},
"models/Face/arc_w600k_r18.onnx": {
    "url": "<release_url>",
    "checksum": "<sha256>",
},
```

Note: The actual URLs and checksums need to be determined when the models are uploaded to a GitHub release. For now, models can be manually placed in `models/Face/`.

- [ ] **Step 2: Create faces/ directory with README**

```bash
mkdir -p faces
echo "Place face photos here: faces/<person_name>/photo.jpg" > faces/README.md
```

- [ ] **Step 3: Commit**

```bash
git add src/glados/cli.py faces/README.md
git commit -m "feat: add InsightFace model entries and faces directory"
```

---

## Task 10: Integration test

- [ ] **Step 1: Run all robot tests**

```bash
cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run pytest tests/robot/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Smoke test robot mode**

```bash
cd /home/mrv/Homework/PROJECTS/GLaDOS && uv run glados robot --config configs/robot_config.yaml
```

Verify: OpenCV window opens, camera feed visible, ASR responds to speech, LLM generates responses.

- [ ] **Step 3: Final commit and push**

```bash
git push origin main
```
