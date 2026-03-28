#!/usr/bin/env python3
"""Minimal Whisper ASR server — OpenAI-compatible transcription endpoint.

Usage on GPU server:
    pip install faster-whisper fastapi uvicorn python-multipart
    python whisper_server.py --model large-v3 --host 0.0.0.0 --port 8000

Endpoint:
    POST /v1/audio/transcriptions
    Form fields: file (audio), language (optional, default "ru")
    Returns: {"text": "transcribed text"}

    GET /health
    Returns: {"status": "ok", "model": "large-v3"}
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper ASR Server")

model: WhisperModel | None = None
model_name: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("ru"),
    model: str = Form(None),  # ignored, uses server's model
):
    assert globals()["model"] is not None, "Model not loaded"

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        segments, _ = globals()["model"].transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        dt = time.perf_counter() - t0
        print(f"[{dt:.2f}s] {text[:100]}")
        return {"text": text}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    global model, model_name

    parser = argparse.ArgumentParser(description="Whisper ASR Server")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = parser.parse_args()

    model_name = args.model
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading Whisper '{args.model}' on {device} ({compute_type})...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    print(f"Model loaded. Server starting on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
