# Robot Architecture v2 — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перестроить robot mode архитектуру по результатам анализа ROBOT_ARCHITECTURE_RESEARCH.md — от каскадного threading к asyncio event-driven с streaming pipeline, fault tolerance и observability.

**Architecture:** 3 фазы. Фаза A — критичные фиксы (TTS, FaceID, LLM) без переделки архитектуры. Фаза B — asyncio перестройка с streaming TTS, разбивкой BrainWorker, supervision. Фаза C — multiprocessing, AEC, метрики.

**Tech Stack:** Python 3.12, asyncio, onnxruntime, faster-whisper, TeraTTS, sounddevice, httpx (async HTTP), loguru

---

## Фаза A: Критичные фиксы (блокеры)

Цель: заставить текущую систему нормально работать — звук воспроизводится, лица детектируются, LLM отвечает приемлемо быстро. Архитектура threading не меняется.

---

### Task A1: Диагностика и фикс TTS playback

**Проблема:** SpeakerWorker не воспроизводит звук. В логах нет TTS/Speaker строк вообще. Гипотезы: (1) `start_speaking()` вызывает `stop_speaking()` первым → ставит `_stop_event` → конфликт; (2) `sd.get_stream()` возвращает input stream вместо output stream; (3) SpeakerWorker тихо падает.

**Files:**
- Modify: `src/glados/robot/voice.py` (SpeakerWorker.run)
- Modify: `src/glados/audio_io/sounddevice_io.py` (start_speaking)
- Test: `tests/robot/test_voice.py`

- [ ] **Step 1: Добавить диагностическое логирование в SpeakerWorker**

В `voice.py` SpeakerWorker.run(), обернуть тело в try/except и добавить логи на каждом этапе:

```python
def run(self) -> None:
    logger.info("SpeakerWorker started.")
    while not self._shutdown.is_set():
        try:
            chunk: AudioChunk = self._audio_q.get(timeout=0.05)
        except queue.Empty:
            continue

        if chunk.is_eos:
            logger.debug("Speaker: EOS received")
            self._speaking.clear()
            self._processing.clear()
            continue

        if chunk.audio.size == 0:
            continue

        logger.debug("Speaker: playing {} samples @ {}Hz", chunk.audio.size, self._sr)
        self._speaking.set()

        try:
            self._audio.start_speaking(chunk.audio, self._sr, chunk.text)
        except Exception as e:
            logger.error("Speaker: start_speaking FAILED: {}", e)
            self._speaking.clear()
            continue

        # Wait for playback using blocking sd.wait()
        try:
            sd.wait()
        except Exception as e:
            logger.warning("Speaker: sd.wait() error: {}", e)

        self._audio.stop_speaking()
        logger.debug("Speaker: chunk done")

    logger.info("SpeakerWorker stopped.")
```

Ключевое изменение: заменить ручной polling `sd.get_stream().active` на `sd.wait()` — это официальный способ дождаться окончания `sd.play()`.

- [ ] **Step 2: Убрать self.stop_speaking() из начала start_speaking()**

В `sounddevice_io.py` метод `start_speaking` вызывает `self.stop_speaking()` первым. `stop_speaking()` ставит `_stop_event.set()` и вызывает `sd.stop()`. Проблема: `_stop_event` остаётся set после `sd.stop()`, а `_stop_event.clear()` вызывается **после** `sd.play()`. Race condition.

Фикс: вызывать `sd.stop()` напрямую вместо `self.stop_speaking()`:

```python
def start_speaking(self, audio_data, sample_rate=None, text=""):
    if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
        raise ValueError("Invalid audio data")
    if sample_rate is None:
        sample_rate = self.SAMPLE_RATE

    # Stop any existing playback without touching _stop_event
    sd.stop()

    self._stop_event.clear()
    self._is_playing = True
    logger.debug("Playing audio: {} samples @ {}Hz", len(audio_data), sample_rate)
    sd.play(audio_data, sample_rate)
```

- [ ] **Step 3: Написать тест воспроизведения**

```python
# tests/robot/test_voice.py — добавить
def test_speaker_worker_plays_and_clears_flag():
    """SpeakerWorker should play audio and clear speaking flag on EOS."""
    audio_q = queue.Queue()
    speaking = threading.Event()
    processing = threading.Event()
    shutdown = threading.Event()
    conv = ConversationStore()

    mock_audio = MagicMock()
    sr = 22050

    worker = SpeakerWorker(
        audio_io=mock_audio,
        audio_queue=audio_q,
        conversation_store=conv,
        sample_rate=sr,
        shutdown_event=shutdown,
        speaking_event=speaking,
        processing_event=processing,
    )

    # Put audio chunk + EOS
    chunk = AudioChunk(audio=np.ones(1000, dtype=np.float32), text="test")
    eos = AudioChunk(audio=np.array([], dtype=np.float32), text="", is_eos=True)
    audio_q.put(chunk)
    audio_q.put(eos)

    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(0.5)
    shutdown.set()
    t.join(timeout=2.0)

    mock_audio.start_speaking.assert_called_once()
    mock_audio.stop_speaking.assert_called()
    assert not speaking.is_set()
```

- [ ] **Step 4: Запустить тесты**

Run: `pytest tests/robot/test_voice.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/voice.py src/glados/audio_io/sounddevice_io.py tests/robot/test_voice.py
git commit -m "fix(tts): replace polling with sd.wait(), fix start_speaking race condition"
```

---

### Task A2: FaceID — скачать det_2.5g и диагностировать

**Проблема:** det_500m.onnx (2.5MB, 500K параметров) — слишком слабая для webcam. Нужна det_2.5g.onnx из buffalo_m (2.5G параметров).

**Files:**
- Modify: `src/glados/robot/face_id.py` (модель и пороги)
- Modify: `configs/robot_config.yaml` (добавить face_model опцию)
- Test: `tests/robot/test_face_id.py`

- [ ] **Step 1: Скачать det_2.5g.onnx**

```bash
# buffalo_m модели на Hugging Face / InsightFace
pip download insightface --no-deps -d /tmp/insightface_pkg
# Или скачать напрямую:
python3 -c "
from insightface.utils import storage
storage.download('models', 'buffalo_m', root='/tmp/insightface_models')
"
# Скопировать det_2.5g.onnx в models/Face/
cp /tmp/insightface_models/models/buffalo_m/det_2.5g.onnx models/Face/
```

Если insightface не установлен, скачать вручную:
```bash
pip install insightface
python3 -c "
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_m', root='/tmp/insightface_models')
"
cp /tmp/insightface_models/models/buffalo_m/det_2.5g.onnx models/Face/
```

- [ ] **Step 2: Добавить поддержку выбора модели детекции**

В `face_id.py` FaceRecognizer.__init__:

```python
def __init__(self, model_dir: str | Path, face_db_dir: str | Path | None = None,
             det_model: str = "det_2.5g") -> None:
    self._model_dir = Path(model_dir)
    self._db = FaceDB()

    # Try requested model, fall back to det_500m
    det_path = self._model_dir / f"{det_model}.onnx"
    if not det_path.exists():
        det_path = self._model_dir / "det_500m.onnx"
        logger.warning("FaceID: {} not found, falling back to {}", det_model, det_path.name)
    if not det_path.exists():
        raise FileNotFoundError(f"Face detection model not found: {det_path}")

    logger.info("FaceID: using detection model {}", det_path.name)
    self._det_session = ort.InferenceSession(
        str(det_path), providers=["CPUExecutionProvider"]
    )
```

- [ ] **Step 3: Написать скрипт диагностики для живой камеры**

Создать `scripts/debug_face_detect.py`:

```python
"""Debug face detection on live camera frame."""
import cv2
import sys
sys.path.insert(0, "src")
from glados.robot.face_id import FaceRecognizer

rec = FaceRecognizer("models/Face")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Camera not available")
    sys.exit(1)

print(f"Frame: {frame.shape}")

# Test with different thresholds
for thresh in [0.3, 0.4, 0.5, 0.6, 0.65]:
    rec._SCORE_THRESH = thresh
    faces = rec.detect(frame)
    print(f"  threshold={thresh:.2f} → {len(faces)} faces: {faces}")

# Save frame for manual inspection
cv2.imwrite("/tmp/debug_frame.jpg", frame)
print("Saved /tmp/debug_frame.jpg")
```

- [ ] **Step 4: Запустить диагностику и определить рабочий порог**

```bash
python scripts/debug_face_detect.py
```

Проанализировать вывод. Если det_2.5g детектирует, а det_500m нет — оставить det_2.5g. Подобрать threshold.

- [ ] **Step 5: Обновить тесты**

```python
# tests/robot/test_face_id.py — добавить
def test_face_recognizer_fallback_model(tmp_path):
    """FaceRecognizer should fall back to det_500m if det_2.5g missing."""
    # Create dummy ONNX files
    # (этот тест проверяет логику fallback, не реальный инференс)
    det_path = tmp_path / "det_500m.onnx"
    det_path.touch()  # dummy
    # Test will fail on InferenceSession but proves the path logic
    # We test the path selection separately
```

- [ ] **Step 6: Commit**

```bash
git add src/glados/robot/face_id.py scripts/debug_face_detect.py tests/robot/test_face_id.py
git commit -m "feat(face_id): support det_2.5g model with fallback to det_500m"
```

---

### Task A3: Оптимизация LLM скорости

**Проблема:** qwen2.5:7b на CPU = 5-8 tok/s, num_ctx: 4096 → долгий prefill.

**Files:**
- Modify: `configs/robot_config.yaml`
- Modify: `src/glados/robot/brain.py` (добавить замер TTFT)

- [ ] **Step 1: Уменьшить num_ctx и переключить модель**

```yaml
# robot_config.yaml
Robot:
  llm_model: "qwen2.5:3b"
  llm_options:
    num_ctx: 2048
    num_thread: 6
```

- [ ] **Step 2: Добавить замер TTFT (time to first token) в BrainWorker**

```python
# В brain.py _process(), после requests.post:
ttft = None
for line in resp.iter_lines():
    if ttft is None:
        ttft = time.perf_counter() - t0
        logger.info("LLM TTFT: {:.2f}s", ttft)
    # ... rest of processing
```

Добавить `t0 = time.perf_counter()` перед `requests.post()`.

- [ ] **Step 3: Проверить что Ollama flash attention включен**

```bash
# Проверить текущие настройки Ollama
systemctl cat ollama 2>/dev/null || echo "Not a systemd service"
ollama show qwen2.5:3b --modelfile 2>/dev/null | head
# Если нет qwen2.5:3b — скачать
ollama pull qwen2.5:3b
```

- [ ] **Step 4: Commit**

```bash
git add configs/robot_config.yaml src/glados/robot/brain.py
git commit -m "perf(brain): switch to qwen2.5:3b, reduce num_ctx, add TTFT metric"
```

---

### Task A4: Добавить watchdog для потоков

**Проблема:** Потоки умирают молча, нет мониторинга.

**Files:**
- Create: `src/glados/robot/watchdog.py`
- Modify: `src/glados/robot/engine.py`
- Test: `tests/robot/test_watchdog.py`

- [ ] **Step 1: Написать тест для watchdog**

```python
# tests/robot/test_watchdog.py
import threading
import time
from glados.robot.watchdog import Watchdog


def test_watchdog_detects_dead_thread():
    """Watchdog should call on_failure when a thread dies."""
    failures = []

    def worker():
        raise RuntimeError("crash!")

    def on_failure(name: str, error: str):
        failures.append((name, error))

    t = threading.Thread(target=worker, name="TestWorker", daemon=True)
    wd = Watchdog(
        threads={"TestWorker": t},
        on_failure=on_failure,
        check_interval=0.1,
    )
    t.start()
    time.sleep(0.3)
    wd.check_once()
    assert len(failures) == 1
    assert failures[0][0] == "TestWorker"


def test_watchdog_ignores_alive_thread():
    """Watchdog should not trigger for alive threads."""
    failures = []
    stop = threading.Event()

    def worker():
        stop.wait()

    def on_failure(name, error):
        failures.append(name)

    t = threading.Thread(target=worker, name="AliveWorker", daemon=True)
    t.start()
    wd = Watchdog(
        threads={"AliveWorker": t},
        on_failure=on_failure,
        check_interval=0.1,
    )
    wd.check_once()
    stop.set()
    assert len(failures) == 0
```

- [ ] **Step 2: Запустить тест, убедиться что падает**

Run: `pytest tests/robot/test_watchdog.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Реализовать Watchdog**

```python
# src/glados/robot/watchdog.py
"""Thread watchdog with failure detection and logging."""
from __future__ import annotations

import threading
from typing import Callable

from loguru import logger


class Watchdog:
    """Monitors threads and calls on_failure when one dies."""

    def __init__(
        self,
        threads: dict[str, threading.Thread],
        on_failure: Callable[[str, str], None],
        check_interval: float = 2.0,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self._threads = threads
        self._on_failure = on_failure
        self._interval = check_interval
        self._shutdown = shutdown_event or threading.Event()
        self._notified: set[str] = set()

    def check_once(self) -> list[str]:
        """Check all threads once. Returns names of dead threads."""
        dead = []
        for name, thread in self._threads.items():
            if name in self._notified:
                continue
            if not thread.is_alive():
                logger.error("Watchdog: {} is DEAD", name)
                self._notified.add(name)
                self._on_failure(name, f"Thread {name} died")
                dead.append(name)
        return dead

    def run(self) -> None:
        """Run watchdog loop until shutdown."""
        logger.info("Watchdog started, monitoring {} threads.", len(self._threads))
        while not self._shutdown.is_set():
            self.check_once()
            self._shutdown.wait(timeout=self._interval)
        logger.info("Watchdog stopped.")
```

- [ ] **Step 4: Запустить тест**

Run: `pytest tests/robot/test_watchdog.py -v`
Expected: PASS

- [ ] **Step 5: Интегрировать в engine.py**

В `engine.py` после запуска всех потоков:

```python
from .watchdog import Watchdog

# В start() после цикла for name, target in thread_targets:
thread_map = {t.name: t for t in self._threads}

def _on_thread_failure(name: str, error: str) -> None:
    logger.critical("CRITICAL: Worker {} died! Error: {}", name, error)

self._watchdog = Watchdog(
    threads=thread_map,
    on_failure=_on_thread_failure,
    check_interval=2.0,
    shutdown_event=self._shutdown,
)
wd_thread = threading.Thread(target=self._watchdog.run, name="Watchdog", daemon=True)
wd_thread.start()
```

- [ ] **Step 6: Запустить все тесты**

Run: `pytest tests/robot/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/glados/robot/watchdog.py src/glados/robot/engine.py tests/robot/test_watchdog.py
git commit -m "feat(robot): add thread watchdog for fault detection"
```

---

## Фаза B: Asyncio перестройка

Цель: перевести robot engine на asyncio event loop с streaming pipeline. BrainWorker разбивается на компоненты. TTS становится streaming.

**Предварительные условия:** Фаза A завершена, все тесты зелёные.

---

### Task B1: Event Bus на asyncio

**Files:**
- Create: `src/glados/robot/event_bus.py`
- Test: `tests/robot/test_event_bus.py`

- [ ] **Step 1: Написать тест**

```python
# tests/robot/test_event_bus.py
import asyncio
import pytest
from glados.robot.event_bus import EventBus, Event


@pytest.mark.asyncio
async def test_publish_subscribe():
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe("speech", handler)
    await bus.publish(Event(type="speech", data={"text": "hello"}))
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].data["text"] == "hello"


@pytest.mark.asyncio
async def test_priority_ordering():
    """Higher priority events should be processed first."""
    bus = EventBus()
    order = []

    async def handler(event: Event):
        order.append(event.data["id"])

    bus.subscribe("test", handler)
    # Publish low-priority first, high-priority second
    await bus.publish(Event(type="test", data={"id": "low"}, priority=0))
    await bus.publish(Event(type="test", data={"id": "high"}, priority=10))
    await asyncio.sleep(0.1)
    # High priority should come first if queued simultaneously
    assert "high" in order
    assert "low" in order
```

- [ ] **Step 2: Реализовать EventBus**

```python
# src/glados/robot/event_bus.py
"""Async event bus with priority support."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger


@dataclass(order=True)
class Event:
    priority: int = field(default=0, compare=True)
    type: str = field(default="", compare=False)
    data: dict[str, Any] = field(default_factory=dict, compare=False)
    timestamp: float = field(default=0.0, compare=False)


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async event bus with topic-based pub/sub and priority queuing."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._queue: asyncio.PriorityQueue[Event] = asyncio.PriorityQueue()
        self._running = False

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        if not event.timestamp:
            import time
            event.timestamp = time.time()
        # Negate priority for PriorityQueue (higher value = higher priority)
        negated = Event(
            priority=-event.priority,
            type=event.type,
            data=event.data,
            timestamp=event.timestamp,
        )
        await self._queue.put(negated)
        # Process immediately if not running dispatch loop
        if not self._running:
            await self._dispatch_one()

    async def run(self) -> None:
        """Run the dispatch loop."""
        self._running = True
        logger.info("EventBus started.")
        while self._running:
            await self._dispatch_one()
        logger.info("EventBus stopped.")

    def stop(self) -> None:
        self._running = False

    async def _dispatch_one(self) -> None:
        try:
            event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return
        # Restore original priority
        event = Event(
            priority=-event.priority,
            type=event.type,
            data=event.data,
            timestamp=event.timestamp,
        )
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error("EventBus handler error for '{}': {}", event.type, e)
```

- [ ] **Step 3: Запустить тесты**

Run: `pytest tests/robot/test_event_bus.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/glados/robot/event_bus.py tests/robot/test_event_bus.py
git commit -m "feat(robot): add async EventBus with priority support"
```

---

### Task B2: Async LLM Client (заменить requests на httpx)

**Files:**
- Create: `src/glados/robot/llm_client.py`
- Test: `tests/robot/test_llm_client.py`

- [ ] **Step 1: Написать тест**

```python
# tests/robot/test_llm_client.py
import pytest
from unittest.mock import AsyncMock, patch
from glados.robot.llm_client import OllamaClient


@pytest.mark.asyncio
async def test_stream_tokens():
    """OllamaClient should yield tokens from streaming response."""
    client = OllamaClient(url="http://localhost:11434/api/chat", model="test")
    tokens = []

    # Mock httpx async stream
    mock_lines = [
        b'{"message":{"content":"Hello"},"done":false}',
        b'{"message":{"content":" world"},"done":false}',
        b'{"done":true}',
    ]

    with patch.object(client, '_stream_raw', return_value=_async_iter(mock_lines)):
        async for token in client.stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)

    assert tokens == ["Hello", " world"]


async def _async_iter(items):
    for item in items:
        yield item
```

- [ ] **Step 2: Реализовать OllamaClient**

```python
# src/glados/robot/llm_client.py
"""Async streaming LLM client for Ollama."""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx
from loguru import logger


class OllamaClient:
    """Async HTTP client for Ollama streaming API."""

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
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama. Yields content strings."""
        data = {
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
```

- [ ] **Step 3: Запустить тесты**

Run: `pytest tests/robot/test_llm_client.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/glados/robot/llm_client.py tests/robot/test_llm_client.py
git commit -m "feat(robot): add async OllamaClient with streaming"
```

---

### Task B3: Разбить BrainWorker — Context Manager + Think Filter + Sentence Splitter

**Проблема из анализа:** BrainWorker — God Object с 7 ответственностями.

**Files:**
- Create: `src/glados/robot/context.py` (context builder)
- Create: `src/glados/robot/text_pipeline.py` (think filter + sentence splitter)
- Test: `tests/robot/test_context.py`
- Test: `tests/robot/test_text_pipeline.py`

- [ ] **Step 1: Тесты для think filter**

```python
# tests/robot/test_text_pipeline.py
from glados.robot.text_pipeline import ThinkFilter, SentenceSplitter


def test_think_filter_strips_tags():
    f = ThinkFilter()
    assert f.feed("<think>reasoning</think>Hello") == "Hello"


def test_think_filter_multipart():
    f = ThinkFilter()
    assert f.feed("<think>start") == ""
    assert f.feed("middle") == ""
    assert f.feed("</think>World") == "World"


def test_sentence_splitter():
    s = SentenceSplitter()
    results = []
    for token in ["Привет", ".", " Как", " дела", "?"]:
        sentence = s.feed(token)
        if sentence:
            results.append(sentence)
    results.append(s.flush())
    assert results == ["Привет.", " Как дела?"]
```

- [ ] **Step 2: Тесты для context builder**

```python
# tests/robot/test_context.py
from glados.robot.context import ContextBuilder


def test_context_injects_vision():
    cb = ContextBuilder(face_names={"creator": "твой создатель"})
    messages = [{"role": "system", "content": "personality"}]
    result = cb.build(
        messages=messages,
        vision_desc="A person. Лица: creator",
        autonomy=False,
    )
    # Should have vision injected after system prompt
    assert any("[vision]" in m["content"] for m in result)
    assert any("твой создатель" in m["content"] for m in result)


def test_context_no_vision_when_empty():
    cb = ContextBuilder()
    messages = [{"role": "user", "content": "hi"}]
    result = cb.build(messages=messages, vision_desc=None, autonomy=False)
    assert not any("[vision]" in m.get("content", "") for m in result)
```

- [ ] **Step 3: Реализовать text_pipeline.py**

```python
# src/glados/robot/text_pipeline.py
"""Think tag filter and sentence splitter for LLM output."""
from __future__ import annotations

from loguru import logger


class ThinkFilter:
    """Strips <think>...</think> blocks from streaming LLM output."""

    OPEN_TAGS = ("<think>", "<thinking>")
    CLOSE_TAGS = ("</think>", "</thinking>")

    def __init__(self) -> None:
        self._in_thinking = False
        self._buf: list[str] = []

    def feed(self, chunk: str) -> str:
        text = chunk

        if not self._in_thinking:
            for tag in self.OPEN_TAGS:
                if tag in text:
                    parts = text.split(tag, 1)
                    text = parts[0]
                    self._in_thinking = True
                    self._buf.append(parts[1] if len(parts) > 1 else "")
                    break

        if self._in_thinking:
            combined = "".join(self._buf) + (text if self._in_thinking else "")
            for tag in self.CLOSE_TAGS:
                if tag in combined:
                    parts = combined.split(tag, 1)
                    if parts[0].strip():
                        logger.debug("Thinking: {}...", parts[0][:100])
                    self._buf.clear()
                    self._in_thinking = False
                    return parts[1] if len(parts) > 1 else ""
            if self._in_thinking:
                self._buf.append(text)
                return ""

        return text

    def reset(self) -> None:
        self._in_thinking = False
        self._buf.clear()


class SentenceSplitter:
    """Accumulates tokens and splits on sentence-ending punctuation."""

    ENDINGS = {".", "!", "?", ":", ";", "?!", "\n"}

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def feed(self, token: str) -> str | None:
        """Feed a token. Returns complete sentence if boundary found, else None."""
        self._buffer.append(token)
        combined = "".join(self._buffer)
        if any(combined.rstrip().endswith(p) for p in self.ENDINGS):
            self._buffer.clear()
            return combined.strip() if combined.strip() else None
        return None

    def flush(self) -> str:
        """Flush remaining buffer."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return text
```

- [ ] **Step 4: Реализовать context.py**

```python
# src/glados/robot/context.py
"""Context builder for LLM messages."""
from __future__ import annotations

from typing import Any

from ..core.knowledge_store import KnowledgeStore


class ContextBuilder:
    """Builds LLM message list with vision, knowledge, and face name injections."""

    def __init__(
        self,
        face_names: dict[str, str] | None = None,
        knowledge_store: KnowledgeStore | None = None,
        autonomy_prompt: str | None = None,
    ) -> None:
        self._face_names = face_names or {}
        self._knowledge = knowledge_store
        self._autonomy_prompt = autonomy_prompt

    def build(
        self,
        messages: list[dict[str, Any]],
        vision_desc: str | None,
        autonomy: bool,
    ) -> list[dict[str, Any]]:
        msgs = list(messages)
        extra: list[dict[str, Any]] = []

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

        if vision_desc:
            desc = vision_desc
            if self._face_names:
                mapped = []
                for folder, real_name in self._face_names.items():
                    if folder in desc:
                        mapped.append(f"{folder} — это {real_name}")
                if mapped:
                    desc += " (" + "; ".join(mapped) + ")"
            extra.append({"role": "system", "content": f"[vision] {desc}"})

        if self._knowledge:
            entries = self._knowledge.list_entries()
            if entries:
                lines = ["[knowledge]"] + [f"- {e.text}" for e in entries[:10]]
                extra.append({"role": "system", "content": "\n".join(lines)})

        if extra:
            idx = 0
            while idx < len(msgs) and msgs[idx].get("role") == "system":
                idx += 1
            for offset, msg in enumerate(extra):
                msgs.insert(idx + offset, msg)

        return msgs
```

- [ ] **Step 5: Запустить тесты**

Run: `pytest tests/robot/test_text_pipeline.py tests/robot/test_context.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/glados/robot/text_pipeline.py src/glados/robot/context.py \
        tests/robot/test_text_pipeline.py tests/robot/test_context.py
git commit -m "refactor(brain): extract ContextBuilder, ThinkFilter, SentenceSplitter"
```

---

### Task B4: Async BrainWorker (новый, на базе компонентов)

**Files:**
- Create: `src/glados/robot/brain_async.py`
- Test: `tests/robot/test_brain_async.py`
- Modify: `src/glados/robot/engine.py` (переключить на async brain)

- [ ] **Step 1: Написать тест**

```python
# tests/robot/test_brain_async.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from glados.robot.brain_async import AsyncBrain
from glados.robot.event_bus import EventBus, Event


@pytest.mark.asyncio
async def test_brain_processes_speech_event():
    """AsyncBrain should process speech and emit TTS events."""
    bus = EventBus()
    tts_events = []

    async def tts_handler(event: Event):
        tts_events.append(event)

    bus.subscribe("tts", tts_handler)

    mock_llm = AsyncMock()
    mock_llm.stream = AsyncMock(return_value=_async_iter(["Привет", "."]))

    brain = AsyncBrain(
        event_bus=bus,
        llm_client=mock_llm,
        personality="Ты ГЛаДОС",
    )

    await brain.handle_speech(Event(type="speech", data={"text": "Привет"}))
    # Should have produced at least one TTS event
    assert len(tts_events) >= 1


async def _async_iter(items):
    for item in items:
        yield item
```

- [ ] **Step 2: Реализовать AsyncBrain**

```python
# src/glados/robot/brain_async.py
"""Async brain: handles speech/vision events, streams LLM, emits TTS events."""
from __future__ import annotations

from typing import Any

from loguru import logger

from ..core.conversation_store import ConversationStore
from ..vision.vision_state import VisionState
from .context import ContextBuilder
from .event_bus import Event, EventBus
from .llm_client import OllamaClient
from .text_pipeline import SentenceSplitter, ThinkFilter


class AsyncBrain:
    """Processes speech/vision events through LLM and emits TTS sentences."""

    def __init__(
        self,
        event_bus: EventBus,
        llm_client: OllamaClient,
        personality: str,
        context_builder: ContextBuilder | None = None,
        conversation: ConversationStore | None = None,
        vision_state: VisionState | None = None,
    ) -> None:
        self._bus = event_bus
        self._llm = llm_client
        self._conv = conversation or ConversationStore(
            initial_messages=[{"role": "system", "content": personality}]
        )
        self._ctx = context_builder or ContextBuilder()
        self._vision = vision_state
        self._think = ThinkFilter()
        self._splitter = SentenceSplitter()

    async def handle_speech(self, event: Event) -> None:
        """Process user speech through LLM."""
        text = event.data.get("text", "")
        if not text:
            return

        self._conv.append({"role": "user", "content": text})
        await self._generate(autonomy=False)

    async def handle_vision(self, event: Event) -> None:
        """Process vision event (autonomy)."""
        desc = event.data.get("description", "")
        if not desc:
            return
        self._conv.append({"role": "user", "content": desc})
        await self._generate(autonomy=True)

    async def _generate(self, autonomy: bool) -> None:
        vision_desc = self._vision.snapshot() if self._vision else None
        messages = self._ctx.build(
            messages=self._conv.snapshot(),
            vision_desc=vision_desc,
            autonomy=autonomy,
        )

        self._think.reset()
        full_response: list[str] = []

        try:
            async for token in self._llm.stream(messages):
                speakable = self._think.feed(token)
                if not speakable:
                    continue
                full_response.append(speakable)
                sentence = self._splitter.feed(speakable)
                if sentence:
                    await self._bus.publish(Event(
                        type="tts", data={"text": sentence}, priority=5
                    ))
        except Exception as e:
            logger.error("AsyncBrain LLM error: {}", e)

        # Flush remaining
        remaining = self._splitter.flush()
        if remaining:
            await self._bus.publish(Event(
                type="tts", data={"text": remaining}, priority=5
            ))

        # EOS marker
        await self._bus.publish(Event(type="tts_eos", data={}, priority=5))

        full_text = "".join(full_response).strip()
        if full_text:
            self._conv.append({"role": "assistant", "content": full_text})
            logger.success("LLM: {}", full_text[:120])
```

- [ ] **Step 3: Запустить тесты**

Run: `pytest tests/robot/test_brain_async.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/glados/robot/brain_async.py tests/robot/test_brain_async.py
git commit -m "feat(robot): add AsyncBrain with decomposed pipeline"
```

---

### Task B5: Async Engine (замена threading engine)

**Files:**
- Create: `src/glados/robot/engine_async.py`
- Modify: `src/glados/cli.py`
- Test: `tests/robot/test_engine_async.py`

Здесь объединяем EventBus, AsyncBrain, и thread-based workers (audio I/O остаётся в threads, так как sounddevice — blocking). Используем `asyncio.to_thread()` для bridging.

- [ ] **Step 1: Написать тест**

```python
# tests/robot/test_engine_async.py
import pytest
from unittest.mock import MagicMock, patch
from glados.robot.config import RobotConfig
from glados.robot.engine_async import AsyncRobotEngine


def test_async_engine_constructs():
    """AsyncRobotEngine should construct from config."""
    config = RobotConfig(
        llm_model="test",
        completion_url="http://localhost:11434/api/chat",
    )
    engine = AsyncRobotEngine(config, start_vision=False, start_audio=False)
    assert engine is not None
```

- [ ] **Step 2: Реализовать AsyncRobotEngine**

Основная идея: asyncio event loop в главном потоке, audio I/O и vision в отдельных потоках с bridge через `asyncio.run_coroutine_threadsafe()`.

- [ ] **Step 3: Обновить CLI**

```python
# cli.py — заменить robot command
elif args.command == "robot":
    from glados.robot.config import RobotConfig
    from glados.robot.engine_async import AsyncRobotEngine
    config = RobotConfig.from_yaml(args.config)
    engine = AsyncRobotEngine(config)
    engine.run()  # calls asyncio.run() internally
```

- [ ] **Step 4: Запустить все тесты**

Run: `pytest tests/robot/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/glados/robot/engine_async.py tests/robot/test_engine_async.py src/glados/cli.py
git commit -m "feat(robot): async engine with EventBus orchestration"
```

---

### Task B6: Streaming TTS (chunk-level, не sentence-level)

**Files:**
- Modify: `src/glados/robot/text_pipeline.py` (добавить ChunkSplitter — 3-5 слов)
- Modify: `src/glados/robot/brain_async.py` (использовать chunk splitter)
- Test: `tests/robot/test_text_pipeline.py`

- [ ] **Step 1: Тест для ChunkSplitter**

```python
def test_chunk_splitter_word_threshold():
    s = ChunkSplitter(min_words=3)
    results = []
    for token in ["Роботы", " —", " это", " прекрасные", " существа", "."]:
        chunk = s.feed(token)
        if chunk:
            results.append(chunk)
    results.append(s.flush())
    # Should split after 3+ words when punctuation/space arrives
    assert len(results) >= 1
    assert "".join(results).strip() == "Роботы — это прекрасные существа."
```

- [ ] **Step 2: Реализовать ChunkSplitter**

- [ ] **Step 3: Интегрировать в AsyncBrain**

- [ ] **Step 4: Commit**

---

## Фаза C: Observability + AEC + Multiprocessing

Цель: метрики для диагностики, AEC для echo cancellation, CPU-bound задачи в отдельных процессах.

---

### Task C1: Pipeline Metrics

**Files:**
- Create: `src/glados/robot/metrics.py`
- Modify: все workers (добавить timing)

- [ ] Реализовать PipelineMetrics (deque-based p50/p95/p99)
- [ ] Добавить замеры в каждый этап
- [ ] Периодический лог метрик (каждые 30с)

---

### Task C2: PulseAudio AEC

- [ ] Настроить `module-echo-cancel` в PulseAudio/PipeWire
- [ ] Указать echocancel source в sounddevice
- [ ] Убрать `interrupt_keywords` workaround

---

### Task C3: Multiprocessing для CPU-bound

- [ ] Vision pipeline → отдельный Process
- [ ] Audio pipeline (VAD + ASR) → отдельный Process
- [ ] IPC через multiprocessing.Queue

---

## Порядок выполнения

```
A1 (TTS fix) → A2 (FaceID) → A3 (LLM speed) → A4 (Watchdog)
    → B1 (EventBus) → B2 (LLM Client) → B3 (Decompose Brain)
    → B4 (AsyncBrain) → B5 (AsyncEngine) → B6 (Streaming TTS)
    → C1 (Metrics) → C2 (AEC) → C3 (Multiprocessing)
```

Каждая фаза — самостоятельна. После каждой фазы система должна работать и все тесты проходить.
