"""BrainWorker: unified LLM processor with priority queues."""
from __future__ import annotations

import json
import queue
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
        face_names: dict[str, str] | None = None,
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
        self._face_names = face_names or {}
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
            item = self._aq.get_nowait()
            # VisionEvent from VisionWorker → convert to dict
            if not isinstance(item, dict):
                return {
                    "content": getattr(item, "description", str(item)),
                    "autonomy": True,
                }
            return item
        except queue.Empty:
            pass
        time.sleep(0.05)
        return None

    def _process(self, req: dict) -> None:
        autonomy = bool(req.get("autonomy"))
        content = req.get("content", "")
        if not content:
            return

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
                    if self._shutdown.is_set() or not self._processing.is_set():
                        break
                    if not line:
                        continue
                    chunk = self._parse_chunk(line)
                    if chunk is None:
                        continue
                    if chunk == "":
                        break

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

        remaining = "".join(sentence_buffer).strip()
        if remaining:
            self._tts.put(remaining)
        self._tts.put("<EOS>")

        full_text = "".join(full_response).strip()
        if full_text:
            self._conv.append({"role": "assistant", "content": full_text})
            logger.success("LLM: {}", full_text[:120])

    def _build_messages(self, autonomy: bool) -> list[dict[str, Any]]:
        messages = self._conv.snapshot()
        extra: list[dict[str, Any]] = []

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

        if self._vision:
            desc = self._vision.snapshot()
            if desc:
                # Inject face name mappings dynamically
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
            while idx < len(messages) and messages[idx].get("role") == "system":
                idx += 1
            for offset, msg in enumerate(extra):
                messages.insert(idx + offset, msg)

        return messages

    def _parse_chunk(self, line: bytes) -> str | None:
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
                combined = "".join(buf) + text
                if tag in combined:
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
