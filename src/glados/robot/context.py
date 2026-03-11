"""Context builder for LLM message list construction."""
from __future__ import annotations

from typing import Any

from loguru import logger


class ContextBuilder:
    """Builds LLM message list with vision, knowledge, and face name injections."""

    def __init__(
        self,
        face_names: dict[str, str] | None = None,
        knowledge_store: Any | None = None,
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
        """Build final message list with context injections."""
        msgs = list(messages)
        extra: list[dict[str, Any]] = []

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

        if vision_desc:
            desc = vision_desc
            if self._face_names:
                mapped = [
                    f"{folder} — это {real_name}"
                    for folder, real_name in self._face_names.items()
                    if folder in desc
                ]
                if mapped:
                    desc += " (" + "; ".join(mapped) + ")"
            extra.append({"role": "system", "content": f"[vision] {desc}"})

        if self._knowledge:
            entries = self._knowledge.list_entries()
            if entries:
                lines = ["[knowledge]"] + [f"- {e.text}" for e in entries[:10]]
                extra.append({"role": "system", "content": "\n".join(lines)})

        if extra:
            # Insert after system messages
            idx = 0
            while idx < len(msgs) and msgs[idx].get("role") == "system":
                idx += 1
            for offset, msg in enumerate(extra):
                msgs.insert(idx + offset, msg)

        return msgs
