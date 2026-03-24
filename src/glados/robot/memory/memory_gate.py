"""Memory gate — decides what to save from conversations."""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from .sqlite_store import SQLiteStore
from .vector_store import VectorStore


class MemoryGate:
    """Decides what to save from conversations into long-term memory.

    Uses heuristics to score conversation turns:
    - Length of exchange (longer = more important)
    - Emotional intensity
    - Personal information disclosed
    - Questions asked about the bot
    - Novel topics (not seen before)

    Saves to both SQLite (structured facts) and ChromaDB (semantic search).
    All writes are synchronous but fast (<50ms for SQLite, <200ms for ChromaDB).
    """

    # Patterns suggesting personal info worth remembering
    _PERSONAL_PATTERNS = [
        re.compile(r"(?:меня зовут|я\s+\w+|мне\s+\d+\s+лет)", re.IGNORECASE),
        re.compile(r"(?:я работаю|я учусь|я люблю|я ненавижу|мне нравится)", re.IGNORECASE),
        re.compile(r"(?:мой\s+\w+|моя\s+\w+|мои\s+\w+)", re.IGNORECASE),
    ]

    # Minimum importance score to save (1-10 scale)
    _MIN_IMPORTANCE = 4

    def __init__(
        self,
        sqlite: SQLiteStore,
        vector: VectorStore,
    ) -> None:
        self._sqlite = sqlite
        self._vector = vector

    def evaluate_and_save(
        self,
        user_text: str,
        assistant_text: str,
        emotion: str = "sarcastic",
        intensity: float = 0.5,
        person: str | None = None,
    ) -> None:
        """Evaluate a conversation turn and save if important enough."""
        importance = self._score_importance(user_text, assistant_text, intensity)

        if importance < self._MIN_IMPORTANCE:
            logger.debug("Memory gate: skip (importance={}/10)", importance)
            return

        # Save to vector store (episodic memory)
        combined = f"User: {user_text}\nGLaDOS: {assistant_text}"
        participants = [person] if person else []
        self._vector.add(
            text=combined,
            importance=importance,
            participants=participants,
        )

        # Log mood
        self._sqlite.log_mood(emotion, intensity, trigger=user_text[:100])

        # Extract and save personal facts
        facts = self._extract_facts(user_text, person)
        for fact in facts:
            self._sqlite.add_fact(
                person=fact["person"],
                subject=fact["subject"],
                predicate=fact["predicate"],
                obj=fact["object"],
                source="conversation",
            )
            logger.info("Fact saved: {} {} {}", fact["subject"], fact["predicate"], fact["object"])

        logger.info(
            "Memory saved (importance={}/10, {} facts): '{}'",
            importance, len(facts), user_text[:60],
        )

    def _score_importance(
        self,
        user_text: str,
        assistant_text: str,
        intensity: float,
    ) -> int:
        """Heuristic importance scoring (1-10)."""
        score = 3  # base

        # Longer exchanges are more important
        total_len = len(user_text) + len(assistant_text)
        if total_len > 200:
            score += 1
        if total_len > 500:
            score += 1

        # High emotional intensity
        if intensity > 0.7:
            score += 1
        if intensity > 0.9:
            score += 1

        # Personal information disclosed
        if any(p.search(user_text) for p in self._PERSONAL_PATTERNS):
            score += 2

        # Questions about the bot (existential, interesting)
        if "?" in user_text and len(user_text) > 30:
            score += 1

        return min(score, 10)

    def _extract_facts(self, user_text: str, person: str | None) -> list[dict[str, str]]:
        """Extract simple facts from user text using patterns."""
        facts: list[dict[str, str]] = []
        person_name = person or "unknown"

        # "Меня зовут X"
        m = re.search(r"меня зовут\s+(\w+)", user_text, re.IGNORECASE)
        if m:
            facts.append({
                "person": person_name,
                "subject": person_name,
                "predicate": "имя",
                "object": m.group(1),
            })

        # "Мне N лет"
        m = re.search(r"мне\s+(\d+)\s+лет", user_text, re.IGNORECASE)
        if m:
            facts.append({
                "person": person_name,
                "subject": person_name,
                "predicate": "возраст",
                "object": m.group(1),
            })

        # "Я работаю / учусь X"
        m = re.search(r"я\s+(работаю|учусь)\s+(.+?)(?:\.|,|$)", user_text, re.IGNORECASE)
        if m:
            facts.append({
                "person": person_name,
                "subject": person_name,
                "predicate": m.group(1).lower(),
                "object": m.group(2).strip(),
            })

        # "Я люблю / ненавижу X"
        m = re.search(r"я\s+(люблю|ненавижу|обожаю)\s+(.+?)(?:\.|,|$)", user_text, re.IGNORECASE)
        if m:
            facts.append({
                "person": person_name,
                "subject": person_name,
                "predicate": m.group(1).lower(),
                "object": m.group(2).strip(),
            })

        return facts
