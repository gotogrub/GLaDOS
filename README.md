# GLaDOS Personality Core

> *"Science isn't about asking why. It's about asking, 'Why not?'"  -  Cave Johnson*

[Русская версия / Russian version](README.ru.md)

A fork of [dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS) with **Russian language support** — a voice AI assistant styled after GLaDOS from Portal. A sarcastic, passive-aggressive artificial intelligence that sees through a camera, hears through a microphone, speaks through a speaker, and judges you accordingly.

## What's added in this fork

- **Russian TTS** — Russian speech synthesis with GLaDOS voice via [TeraTTS](https://github.com/Tera2Space/TeraTTS) + [ruaccent](https://github.com/Den4ikAI/ruaccent) for proper stress placement
- **Russian ASR** — Whisper-based speech recognition with Russian language support via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Russian config** — ready-to-use `configs/glados_config_ru.yaml` with Russian system prompt and few-shot examples in GLaDOS style
- **Smart interruptions** — `interrupt_keywords` config to prevent GLaDOS from interrupting herself; only reacts to specific words like "замолчи", "стоп"
- **Tools toggle** — `tools_enabled: false` allows using models without function calling support (e.g. Gemma 3)
- **Vision frame saving** — `save_frames: true` saves camera snapshots with VLM descriptions to disk
- **LLM options passthrough** — `llm_options` config field to tune Ollama parameters (`num_ctx`, `num_thread`, etc.)
- **Lazy ASR loading** — speech recognition model loads only on first unmute, speeding up startup
- **File logging** — per-run log files in `logs/YYYY-MM-DD_runNN.log` with full DEBUG detail
- **Lite config** — `configs/glados_config_ru_lite.yaml` optimized for mini-PCs (qwen2.5:3b, CTC ASR, reduced context window)

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Install Ollama: https://github.com/ollama/ollama
ollama pull gemma3:4b       # fast, no tool support
# or
ollama pull qwen2.5:7b      # slower, supports tools + good Russian
```

### 2. Clone and install

```bash
git clone https://github.com/gotogrub/GLaDOS.git
cd GLaDOS
python scripts/install.py
uv pip install -e ".[cpu,ru,tui]"
```

### 3. Download models

```bash
uv run glados download
```

### 4. Run

```bash
uv run glados tui --config configs/glados_config_ru.yaml
```

## Run modes

```bash
# Russian version (TUI)
uv run glados tui --config configs/glados_config_ru.yaml

# English version (original)
uv run glados tui

# Voice mode (English ASR)
uv run glados start

# Text only
uv run glados start --input-mode text --config configs/glados_config_ru.yaml

# Speak a phrase
uv run glados say "The cake is a lie"

# Robot mode (minimal 5-thread engine, no TUI)
uv run glados robot --config configs/robot_config.yaml
```

### Robot Mode (async engine)

Event-driven async engine for robotics — no TUI, no MCP, just speech + vision + LLM with streaming TTS:

```bash
uv run glados robot --config configs/robot_config.yaml
# or
python -m glados robot --config configs/robot_config.yaml
```

Architecture:
- **Async EventBus** — pub/sub in the main asyncio loop (speech, vision, tts, tts_eos events)
- **AsyncBrain** — streams LLM tokens, emits TTS chunks mid-sentence (4-word threshold) for low latency
- **Thread workers** — SpeechWorker, VisionWorker, SpeakerWorker run blocking I/O in daemon threads
- **VoiceLoop** — async TTS synthesis task bridges text→audio via `asyncio.to_thread`
- **Watchdog** — monitors worker threads, logs failures

Features:
- **Face recognition** via SCRFD + ArcFace (ONNX) with per-face profiles and descriptions
- **3 OpenCV windows** — live camera feed, analyzed snapshots with face bboxes, VLM text panel (Cyrillic via PIL)
- **Streaming TTS** — speech starts before the full sentence is generated
- **Conversation trimming** — keeps last 10 turns to prevent context growth and TTFT degradation
- **Per-run file logging** — `logs/YYYY-MM-DD_runNN.log` with DEBUG level

#### Adding faces

```
faces/
  creator/
    photo1.jpg
    photo2.jpg
  alice/
    photo1.jpg
```

Photos are indexed at startup. Configure face profiles in `robot_config.yaml`:

```yaml
face_names:
  creator:
    name: "Creator"
    description: "Your creator. A young man with glasses. Treat him respectfully, but in your sarcastic style."
  alice:
    name: "Alice"
    description: "A colleague. She likes cats."
```

The description is injected into LLM context when the face is recognized, so GLaDOS knows *who* she's talking to.

#### Roadmap

- **Phase A** (done): Critical fixes — TTS playback, FaceID detection, sd.wait()
- **Phase B** (done): Async architecture — EventBus, AsyncBrain, streaming TTS, Watchdog
- **Phase C** (planned): PipelineMetrics, PulseAudio AEC, multiprocessing for CPU-bound tasks
- **Phase D** (planned): Emotion face display — LLM emotion tags → fullscreen images on a monitor (GLaDOS "face")
- **Phase E** (planned): Motor control via ToolExecutor (GPIO, serial), obstacle sensors
- **Phase F** (planned): Navigation (SLAM), path planning, autonomous movement

## Configuration

Config files:
- `configs/glados_config.yaml` — English (original)
- `configs/glados_config_ru.yaml` — Russian (full features)
- `configs/glados_config_ru_lite.yaml` — Russian (lightweight, for mini-PCs)

### Voices

Set `voice` in config:

| Value | Language | Description |
|-------|----------|-------------|
| `glados` | EN | Original GLaDOS voice |
| `glados_ru` | RU | Russian GLaDOS voice (TeraTTS) |
| `af_bella`, `am_adam`, ... | EN | Kokoro voices |

### Changing the LLM

```yaml
llm_model: "gemma3:4b"        # fast, good Russian (no tools)
# llm_model: "qwen2.5:7b"     # good Russian + tool support
# llm_model: "qwen2.5:3b"     # lightweight
# llm_model: "llama3.2"       # English only
```

Models without tool support require `tools_enabled: false` in config.

Browse models: [ollama.com/library](https://ollama.com/library)

### LLM Performance Tuning

Pass Ollama options directly via config:

```yaml
llm_options:
  num_ctx: 2048      # context window (lower = faster)
  num_thread: 8      # CPU threads (half of total is a good default)
```

For optimal Ollama performance on CPU, configure the systemd service:

```bash
sudo systemctl edit ollama.service
```

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_FLASH_ATTENTION=1"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

### ASR (Speech Recognition)

Three ASR engines available via `asr_engine`:

| Engine | Language | Size | Speed |
|--------|----------|------|-------|
| `whisper` | Multilingual (RU, EN, ...) | ~140MB | Medium |
| `ctc` | English only | ~110MB | Fast |
| `tdt` | English only | ~600MB | Best accuracy |

Russian config uses `whisper` by default.

### Interruption Control

```yaml
interruptible: true
# Without keywords: any speech interrupts GLaDOS (causes self-interruption via speakers)
# With keywords: only specific phrases interrupt
interrupt_keywords: ["замолчи", "заткнись", "стоп", "хватит", "тихо", "shut up", "stop"]
```

### Vision (Computer Vision)

```yaml
vision:
  camera_index: 0
  capture_interval_seconds: 5
  scene_change_threshold: 0.05
  max_tokens: 64
  save_frames: true              # save snapshots to disk
  save_frames_dir: "vision_frames"
  save_frames_max: 1000
```

When enabled, GLaDOS sees through the camera and injects scene descriptions into LLM context. With autonomy enabled, she reacts to scene changes automatically.

### Custom Personality

```yaml
personality_preprompt:
  - system: "You are a sarcastic AI who judges humans."
  - user: "What do you think of my code?"
  - assistant: "I've seen better output from a random number generator."
```

### MCP Servers

```yaml
mcp_servers:
  - name: "system_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.system_info_server"]
```

Built-in: `system_info`, `time_info`, `disk_info`, `network_info`, `process_info`, `power_info`, `memory`

Requires `tools_enabled: true` (default). Details: [docs/mcp.md](/docs/mcp.md)

## TUI Controls

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Command palette |
| `F1` | Help |
| `Ctrl+D` | Dialog panel |
| `Ctrl+L` | Logs panel |
| `Ctrl+S` | Status panel |
| `Ctrl+A` | Autonomy panel |
| `Ctrl+U` | Queue panel |
| `Ctrl+M` | MCP panel |
| `Ctrl+I` | Toggle right panels |
| `Ctrl+R` | Restore all panels |

## Architecture

### TUI Mode (full agent)

```mermaid
flowchart TB
    subgraph Input
        mic[Microphone] --> vad[VAD] --> asr[ASR]
        text[Text Input]
        tick[Timer]
        cam[Camera]--> vlm[VLM]
    end

    subgraph Minds["Subagents"]
        sensors[Sensors]
        weather[Weather]
        emotion[Emotion]
        news[News]
        memory[Memory]
    end

    ctx[Context]

    subgraph Core["Main Agent"]
        llm[LLM]
        tts[TTS]
    end

    subgraph Output
        speaker[Speaker]
        logs[Logs]
    end

    asr -->|priority| llm
    text -->|priority| llm
    vlm --> ctx
    tick -->|autonomy| llm

    Minds -->|write| ctx
    ctx -->|read| llm
    llm --> tts --> speaker
    llm --> logs
    llm <-->|MCP| tools[Tools]
```

### Robot Mode (async engine)

```mermaid
flowchart LR
    subgraph Threads
        mic[Mic] --> vad[VAD] --> asr[ASR]
        cam[Camera] --> vlm[VLM]
        vlm --> face[FaceID]
    end

    subgraph "AsyncIO EventBus"
        asr -->|speech event| brain[AsyncBrain]
        vlm -->|vision event| brain
        face -->|face profiles| ctx[ContextBuilder]
        ctx --> brain
        brain -->|tts event| chunk[ChunkSplitter]
        chunk -->|stream| voice[VoiceLoop]
    end

    subgraph Playback
        voice --> speaker[SpeakerWorker]
    end

    brain <-->|httpx stream| ollama[Ollama LLM]
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| ASR (EN) | Parakeet TDT/CTC (ONNX) | English speech recognition |
| ASR (RU) | faster-whisper (CTranslate2) | Multilingual speech recognition |
| VAD | Silero VAD (ONNX) | Voice activity detection |
| TTS (EN) | Kokoro / GLaDOS | English speech synthesis |
| TTS (RU) | TeraTTS + ruaccent | Russian speech synthesis |
| Vision | FastVLM (ONNX) | Scene understanding |
| LLM | OpenAI-compatible API | Reasoning, tool use |
| Tools | MCP Protocol | Extensibility |
| Memory | MCP + Subagent | Long-term memory |
| Emotions | PAD + HEXACO | Emotional state |

## Russian Language Dependencies

```bash
uv pip install -e ".[cpu,ru,tui]"
```

Installs:
- **TeraTTS** — VITS model for Russian TTS (model `TeraTTS/glados2-g2p-vits` downloads automatically from HuggingFace on first run)
- **ruaccent** — Russian text stress placement
- **faster-whisper** — Whisper ASR with Russian support (model downloads on first run)

## Troubleshooting

**GLaDOS responds to herself:**
Set `interrupt_keywords` to only allow specific interrupt phrases. Or use headphones / `interruptible: false`.

**Slow startup:**
ASR model loads at startup. Use `asr_muted: true` to defer loading until `/asr on`.

**Model doesn't support tools:**
Set `tools_enabled: false` for models like Gemma 3 that don't support function calling.

**Russian TTS not working:**
Make sure dependencies are installed: `uv pip install -e ".[cpu,ru,tui]"`

**Logs:**
Robot mode writes per-run logs to `logs/YYYY-MM-DD_runNN.log` (DEBUG level). TUI mode logs to stderr.

**Windows DLL error:**
Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Credits

- [dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS) — original project
- [TeraTTS](https://github.com/Tera2Space/TeraTTS) — Russian TTS
- [ruaccent](https://github.com/Den4ikAI/ruaccent) — Russian text accentuation
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Multilingual ASR
- [KPEKEP/GLaDOS](https://github.com/KPEKEP/GLaDOS) — inspiration for Russian integration
- [Ollama](https://ollama.ai/) — local LLM inference

## License

Original project license. See [LICENSE.txt](LICENSE.txt).
