# GLaDOS Fork — Architecture Review & Robotics Roadmap

## Current Architecture

### Pipeline Overview

```
Input Layer          Processing Layer        Output Layer
─────────────       ──────────────────      ─────────────
Microphone ──┐                              ┌── Speaker
  VAD ───────┤      ┌─────────────────┐     │
  ASR ───────┼──→   │  LLM Processor  │ ──→ ├── TTS
Text Input ──┤      │  (Ollama API)   │     │
Camera ──────┤      └────────┬────────┘     └── Logs
  FastVLM ───┘               │
                        MCP Tools
                    (memory, sensors, etc.)
```

### Components

| Component | File | Tech | Weight |
|-----------|------|------|--------|
| VAD | `audio_io/vad.py` | Silero VAD (ONNX) | ~2MB |
| ASR (EN) | `ASR/ctc_asr.py`, `ASR/tdt_asr.py` | Parakeet (ONNX) | 110-600MB |
| ASR (RU) | `ASR/whisper_asr.py` | faster-whisper (CTranslate2) | ~140MB |
| TTS (EN) | `TTS/tts_glados.py`, `TTS/tts_kokoro.py` | VITS/Kokoro (ONNX) | ~80MB |
| TTS (RU) | `TTS/tts_teratts.py` | TeraTTS + ruaccent | ~100MB |
| Vision | `vision/fastvlm.py` | FastVLM (ONNX) | ~650MB |
| LLM | `core/llm_processor.py` | Ollama / OpenAI API | External |
| Tools | `mcp/*.py` | MCP Protocol | Runtime |
| Autonomy | `autonomy/loop.py` | Event-driven | Runtime |
| Emotions | `autonomy/agents/emotion_agent.py` | PAD + HEXACO model | Runtime |
| Memory | `mcp/memory_server.py` | MCP + file store | Runtime |
| Engine | `core/engine.py` | Threading orchestrator | Runtime |
| TUI | `tui.py` | Textual framework | Runtime |

### Threading Model

- **SpeechListener** — reads mic, VAD, ASR (daemon)
- **TextListener** — reads stdin (daemon)
- **LLMProcessor (priority)** — handles user requests
- **LLMProcessor (autonomy)** — handles autonomous reactions
- **VisionProcessor** — captures frames, runs FastVLM
- **TTSSynthesizer** — converts text to audio
- **SpeechPlayer** — plays audio output
- **ToolExecutor** — runs MCP tool calls
- **AutonomyLoop** — processes events from vision/timer

Total: ~9 concurrent threads + MCP subprocesses.

### What We Changed in This Fork

1. **Russian TTS** (`tts_teratts.py`) — TeraTTS + ruaccent with monkey-patch for transformers>=5.x
2. **Whisper ASR** (`whisper_asr.py`) — faster-whisper for Russian speech recognition
3. **Smart interruptions** — keyword-based interrupt filtering in `speech_listener.py`
4. **Tools toggle** (`tools_enabled`) — disable function calling for models that don't support it
5. **LLM options passthrough** (`llm_options`) — direct Ollama parameter tuning
6. **Vision frame saving** — save camera snapshots with descriptions
7. **File logging** — `glados.log` via loguru in TUI
8. **Lazy ASR loading** — defer heavy model loading until first unmute
9. **LLM timeout** — increased from 30s to 120s for CPU inference

## Robot Mode (New)

A minimal 5-thread engine (`src/glados/robot/`) designed for robotics use cases. Replaces the 10-thread TUI-oriented core with a focused pipeline:

```
Microphone → VAD → ASR ──→ BrainWorker (priority queue) ──→ VoiceWorker → SpeakerWorker → Speaker
Camera → FastVLM + FaceID ──→ BrainWorker (autonomy queue) ↗
```

### Robot Mode Architecture

| Thread | Module | Responsibility |
|--------|--------|---------------|
| SpeechWorker | `robot/speech.py` | Mic → VAD → ASR → LLM queue |
| VisionWorker | `robot/vision.py` | Camera → FastVLM + InsightFace → events + cv2 display |
| BrainWorker | `robot/brain.py` | Unified LLM processor with priority queues |
| VoiceWorker | `robot/voice.py` | TTS synthesis |
| SpeakerWorker | `robot/voice.py` | Audio playback |

### Key Differences from TUI Mode

| Aspect | TUI Mode | Robot Mode |
|--------|----------|------------|
| Threads | ~9 + MCP subprocesses | 5 (no MCP, no autonomy loop, no text input) |
| LLM Processors | 2 (priority + autonomy) | 1 unified BrainWorker |
| Face Recognition | None | InsightFace ONNX (SCRFD + ArcFace) |
| Display | Textual TUI | OpenCV window with face bboxes |
| Context Injections | 6 per request | 3 (vision, knowledge, autonomy) |
| Config | `glados_config*.yaml` | `robot_config.yaml` (Pydantic model) |

### Face Recognition

- **SCRFD** — face detection (~3MB ONNX)
- **ArcFace r18** — face embedding (~60MB ONNX)
- **FaceDB** — in-memory cosine similarity matching
- Photos indexed from `faces/<name>/` at startup

## Robotics Roadmap — How to Make a Full Robot

### Phase 1: Embodied Assistant (Software Only)

**Goal:** GLaDOS as a stationary smart home controller with camera and mic.

Already implemented:
- Vision (camera + scene understanding)
- Voice I/O (ASR + TTS)
- Autonomy (reacts to environment changes)
- Memory (MCP server)
- Emotion model

Missing:
- **Home Assistant MCP** — connect to smart home devices (config already has placeholder)
- **Better Russian ASR** — Whisper base is okay, but `small` or `medium` would be more accurate
- **Wake word detection** — existing `wake_word` config works with Levenshtein distance, needs Russian wake word
- **Echo cancellation** — software-based AEC to prevent self-interruption without keywords hack

### Phase 2: Mobile Robot Platform

**Goal:** GLaDOS on wheels with sensors.

Hardware needed:
- Raspberry Pi 5 / Jetson Nano / mini-PC (16GB RAM)
- USB camera (already supported)
- USB microphone + speaker
- Motor controller (via GPIO/serial)
- Optional: LIDAR, ultrasonic sensors, IMU

Software additions:
- **Motor MCP server** — new MCP server for movement commands (forward, turn, stop)
- **Navigation MCP server** — pathfinding, obstacle avoidance
- **Sensor MCP server** — distance sensors, battery level, IMU data
- **SLAM integration** — mapping environment (ROS2 bridge via MCP)

Architecture change:
```yaml
mcp_servers:
  - name: "motors"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.motor_server"]
  - name: "navigation"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.navigation_server"]
  - name: "sensors"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.sensor_server"]
```

The LLM decides when and how to move via tool calls — no hardcoded behavior.

### Phase 3: Full Humanoid / Advanced Robot

**Goal:** Expressive robot with arms, face, or display.

Additions:
- **Expression display** — LED matrix or screen showing GLaDOS eye/emotions
- **Servo control MCP** — pan-tilt camera head, arms
- **Object manipulation** — pick up objects with gripper
- **Multi-camera** — depth camera (RealSense) for 3D understanding

## Who Would Benefit

| Audience | Use Case |
|----------|----------|
| **Hobbyists / Makers** | Fun robot project with personality, runs locally |
| **Smart home users** | Voice control for Home Assistant in Russian |
| **Educators** | Teaching AI/robotics concepts with engaging personality |
| **Accessibility** | Voice interface for people who can't use screens |
| **Researchers** | Multimodal AI agent testbed (vision + voice + tools + autonomy) |
| **Content creators** | Interactive streaming bot, YouTube/Twitch companion |
| **Elderly care** | Companion robot with personality (less clinical than Alexa) |

## Key Advantages of This Architecture

1. **Fully local** — no cloud, no subscriptions, no data leaves the device
2. **Modular** — swap any component (ASR, TTS, LLM, Vision) independently
3. **MCP extensibility** — add new capabilities without touching core code
4. **Personality-driven** — configurable via system prompt, not hardcoded
5. **Multilingual** — Russian and English with framework for more languages
6. **Resource-aware** — lite config for constrained hardware, full config for desktop

## Bottlenecks & Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| LLM inference speed on CPU | 8-15 tok/s with 3-7B models | Use smaller models, reduce num_ctx |
| Whisper ASR latency | ~2-3s per utterance | Use streaming ASR (not yet implemented) |
| No echo cancellation | Self-interruption via speakers | interrupt_keywords workaround |
| FastVLM English-only descriptions | Vision descriptions in English even for Russian config | Fine-tune or use multilingual VLM |
| Single Ollama instance | Autonomy and user compete for LLM | OLLAMA_NUM_PARALLEL or separate instances |
| No GPU acceleration | All inference on CPU | Jetson Nano / iGPU offloading potential |
