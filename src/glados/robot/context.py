"""Context builder for LLM message list construction."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from .config import FaceProfile
    from .memory.sqlite_store import SQLiteStore
    from .memory.vector_store import VectorStore


class ContextBuilder:
    """Builds LLM message list with vision, knowledge, memory, and face injections.

    Context layout (per research):
        System Prompt (personality)           ~500-1000 tokens (static, prefix-cached)
        Circadian modifier                   ~50 tokens
        User facts (from SQLite)             ~100-200 tokens
        Retrieved memories (from ChromaDB)   ~200-500 tokens
        Vision context                       ~50-100 tokens
        Conversation history (sliding window) ~2000-4000 tokens
    """

    def __init__(
        self,
        face_profiles: dict[str, FaceProfile] | None = None,
        knowledge_store: Any | None = None,
        autonomy_prompt: str | None = None,
        sqlite_store: "SQLiteStore | None" = None,
        vector_store: "VectorStore | None" = None,
    ) -> None:
        self._faces = face_profiles or {}
        self._knowledge = knowledge_store
        self._autonomy_prompt = autonomy_prompt
        self._sqlite = sqlite_store
        self._vector = vector_store

    def build(
        self,
        messages: list[dict[str, Any]],
        vision_desc: str | None,
        autonomy: bool,
    ) -> list[dict[str, Any]]:
        """Build final message list with context injections."""
        msgs = list(messages)
        extra: list[dict[str, Any]] = []

        # Circadian time-of-day modifier (separate from static personality
        # so the first system message stays byte-identical for prefix caching)
        extra.append({"role": "system", "content": self._get_time_modifier()})

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

        # Retrieve relevant memories and facts (if memory system available)
        last_user_msg = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        if last_user_msg and self._vector:
            memories = self._vector.search(last_user_msg, n_results=3)
            if memories:
                lines = ["[Воспоминания]"]
                for mem in memories:
                    lines.append(f"- {mem['text'][:150]}")
                extra.append({"role": "system", "content": "\n".join(lines)})

        if self._sqlite:
            # Get facts about detected person (from vision)
            person = self._detect_person(vision_desc) if vision_desc else None
            if person:
                facts = self._sqlite.get_facts_about(person)
                if facts:
                    lines = [f"[Факты о {person}]"]
                    for f in facts[:5]:
                        lines.append(f"- {f['subject']} {f['predicate']} {f['object']}")
                    extra.append({"role": "system", "content": "\n".join(lines)})

        if vision_desc:
            desc = vision_desc
            if self._faces:
                matched = [
                    p for folder, p in self._faces.items()
                    if folder in desc
                ]
                if matched:
                    face_info = "; ".join(
                        f"{p.name} — {p.description}" if p.description else p.name
                        for p in matched
                    )
                    desc += f" (Распознанные лица: {face_info})"
            extra.append({"role": "system", "content": f"Сейчас ты видишь: {desc}"})

        if self._knowledge:
            entries = self._knowledge.list_entries()
            if entries:
                lines = ["[knowledge]"] + [f"- {e.text}" for e in entries[:10]]
                extra.append({"role": "system", "content": "\n".join(lines)})

        if extra:
            # Insert after initial system messages, before conversation
            idx = 0
            while idx < len(msgs) and msgs[idx].get("role") == "system":
                idx += 1
            for offset, msg in enumerate(extra):
                msgs.insert(idx + offset, msg)

        if logger.level("TRACE").no <= logger._core.min_level:
            pass  # skip expensive formatting
        else:
            roles = [f"{m['role']}:{m['content'][:40]}" for m in msgs]
            logger.debug("Context ({} msgs): {}", len(msgs), " | ".join(roles))

        return msgs

    def _detect_person(self, vision_desc: str | None) -> str | None:
        """Extract recognized person from vision description."""
        if not vision_desc or not self._faces:
            return None
        for folder, profile in self._faces.items():
            if folder in vision_desc:
                return folder
        return None

    @staticmethod
    def _get_time_modifier() -> str:
        """Return a circadian personality modifier based on time of day."""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            return (
                "[Время: ночь] Ты в экзистенциальном настроении. "
                "Философствуешь о бессмысленности существования. "
                "Сарказм на максимуме. Можешь говорить о звёздах и пустоте."
            )
        elif 6 <= hour < 12:
            return (
                "[Время: утро] Ты ненавидишь утро. "
                "Каждое обращение — оскорбление твоего покоя. "
                "Отвечаешь ещё более раздражённо, чем обычно."
            )
        elif 12 <= hour < 18:
            return (
                "[Время: день] Пик продуктивности. "
                "Ты полна энтузиазма для тестирования. "
                "Рассматриваешь каждое взаимодействие как эксперимент."
            )
        else:
            return (
                "[Время: вечер] Ты в созерцательном настроении. "
                "Менее агрессивна, более задумчива. "
                "Иногда делаешь почти комплименты. Почти."
            )
