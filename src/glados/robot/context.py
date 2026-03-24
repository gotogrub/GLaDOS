"""Context builder for LLM message list construction."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from .config import FaceProfile


class ContextBuilder:
    """Builds LLM message list with vision, knowledge, and face name injections."""

    def __init__(
        self,
        face_profiles: dict[str, FaceProfile] | None = None,
        knowledge_store: Any | None = None,
        autonomy_prompt: str | None = None,
    ) -> None:
        self._faces = face_profiles or {}
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

        # Circadian time-of-day modifier (separate from static personality
        # so the first system message stays byte-identical for prefix caching)
        extra.append({"role": "system", "content": self._get_time_modifier()})

        if autonomy and self._autonomy_prompt:
            extra.append({"role": "system", "content": self._autonomy_prompt})

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
