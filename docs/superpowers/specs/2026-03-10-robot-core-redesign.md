# Robot Core Redesign

## Summary

Rebuild GLaDOS core as a minimal 5-thread engine optimized for a mobile robot platform. Add face recognition via InsightFace ONNX. Replace TUI with OpenCV debug window. Keep knowledge base and tools for future expansion.

## Architecture

```
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐    ┌─────────┐
│   Mic   │───▶│ SpeechWorker │───▶│ BrainWorker │───▶│VoiceWorker│───▶│ Speaker │
└─────────┘    │ (VAD + ASR)  │    │ (LLM, prio  │    │  (TTS)    │    └─────────┘
               └──────────────┘    │  + autonomy) │    └───────────┘
                                   └──────┬───────┘
┌─────────┐    ┌──────────────┐           │
│ Camera  │───▶│ VisionWorker │───────────┘
└─────────┘    │ (VLM + Face  │
               │  + cv2 show) │
               └──────────────┘
```

### Threads (5 total)

| Thread | Responsibility |
|--------|---------------|
| SpeechWorker | Mic → VAD → Whisper ASR → priority queue |
| VisionWorker | Camera → FastVLM + InsightFace → VisionState + EventBus + cv2.imshow |
| BrainWorker | Unified LLM processor: priority queue first, then autonomy. Streams to TTS queue |
| VoiceWorker | TTS synthesis: text → audio samples |
| SpeakerWorker | Audio playback via sounddevice |

### Removed components

| Component | Reason |
|-----------|--------|
| TextListener | Robot has no keyboard |
| ContextBuilder | Direct injection in BrainWorker |
| ObservabilityBus + MindRegistry | Replaced by loguru |
| ShutdownOrchestrator | Simple shutdown_event.set() |
| ConstitutionalState | Not needed for robot |
| PreferencesStore | Not needed |
| MemoryContext + MCP memory_server | Inline if needed later |
| InteractionState | Timestamps directly in BrainWorker |
| Command registry (20+ commands) | Removed |
| TUI (1460 lines) | Replaced by cv2 window |

### Kept components

| Component | Reason |
|-----------|--------|
| KnowledgeStore | Knowledge base for LLM context |
| ToolExecutor | Future: motor control, GPIO, sensors |
| ConversationStore | Dialog history |
| VisionState | Thread-safe scene description |
| whisper_asr.py | Russian ASR |
| fastvlm.py | Scene description |
| TTS (glados_ru) | Speech synthesis |

## Vision Pipeline

```
Camera frame (every 3-5s)
    │
    ├──▶ FastVLM (ONNX) ──▶ scene description
    │
    ├──▶ InsightFace (ONNX):
    │       ├── SCRFD face detection (~3MB)
    │       ├── ArcFace embedding (r18, ~60MB)
    │       └── Compare with faces/ DB ──▶ names + distances
    │
    └──▶ Combine ──▶ "[vision] scene desc. Faces: Maxim"
                          │
                          ├──▶ VisionState
                          ├──▶ EventBus → BrainWorker autonomy
                          └──▶ cv2.imshow (bbox + names + scene text)
```

### Face database

```
faces/
  maxim/
    photo1.jpg
    photo2.jpg
  alice/
    photo1.jpg
```

On startup: scan faces/*/, detect → crop → ArcFace embedding, average per person. Store as `dict[str, NDArray]`. Matching via cosine distance, threshold ~0.4.

### InsightFace models

- `models/Face/det_scrfd_2.5g.onnx` (~3MB) — face detection
- `models/Face/arc_w600k_r18.onnx` (~60MB) — ArcFace embeddings (r18 for CPU speed)

## BrainWorker

Single thread, two input queues with priority:

1. Check priority queue (user speech) — non-blocking
2. Check autonomy queue (vision events) — non-blocking
3. If nothing, sleep 50ms

### LLM context (minimal):

1. System prompt (personality + vision instructions) — ~100 tokens
2. `[vision]` scene + faces — ~40 tokens
3. `[knowledge]` if entries exist — ~50 tokens
4. Dialog history — remaining context

### Autonomy mode

No speak/do_nothing tools. Text response goes directly to TTS. Empty response = silence.

## File Structure

```
src/glados/
  robot/
    __init__.py
    engine.py       # RobotEngine — starts 5 threads, ~200 lines
    brain.py        # BrainWorker — unified LLM, ~300 lines
    speech.py       # SpeechWorker — mic→ASR, ~150 lines
    voice.py        # VoiceWorker + SpeakerWorker, ~150 lines
    vision.py       # VisionWorker — VLM + FaceID + cv2 display, ~250 lines
    face_id.py      # InsightFace ONNX wrapper, ~120 lines
    config.py       # RobotConfig (pydantic), ~50 lines
```

Entry point: `glados robot --config configs/robot_config.yaml`

## Config

```yaml
Robot:
  llm_model: "gemma3:4b"
  completion_url: "http://localhost:11434/api/chat"
  llm_options:
    num_ctx: 4096
    num_thread: 8
  asr_engine: "whisper"
  voice: "glados_ru"
  vision:
    camera_index: 0
    capture_interval_seconds: 3
    scene_change_threshold: 0.05
    max_tokens: 64
  face_db: "faces/"
  knowledge_path: "data/knowledge.json"
  personality: "Ты — ГЛаДОС из Portal. Саркастичный ИИ. Отвечай кратко, на русском. Комментируй что видишь."
  autonomy:
    cooldown_s: 5
    tick_prompt: "Сцена: {scene}\nЛица: {faces}"
```

## Metrics

- Old: 10 threads, ~4000 lines (engine+tui+llm_processor), 6 context injections
- New: 5 threads, ~1220 lines, 3 context injections
- Target platforms: Raspberry Pi 5, Jetson Nano, mini-PC

## Roadmap

- Phase 1 (current): Vision + FaceID + minimal engine
- Phase 2: Motor control via ToolExecutor (GPIO, serial), obstacle sensors
- Phase 3: Navigation (SLAM), path planning, autonomous movement
