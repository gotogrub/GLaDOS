"""SQLite storage for conversations, facts, mood, and diary entries."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class SQLiteStore:
    """Structured memory storage backed by SQLite.

    Tables:
    - conversations: session-level metadata
    - messages: individual messages within conversations
    - user_facts: extracted facts about people (subject-predicate-object)
    - mood_log: periodic mood/emotion snapshots
    - diary_entries: daily consolidation / inner monologue
    """

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("SQLiteStore opened: {}", self._path)

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS user_facts (
                id TEXT PRIMARY KEY,
                person TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                source TEXT,
                created_at TEXT NOT NULL,
                superseded_by TEXT,
                FOREIGN KEY (superseded_by) REFERENCES user_facts(id)
            );

            CREATE TABLE IF NOT EXISTS mood_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                intensity REAL DEFAULT 0.5,
                trigger TEXT
            );

            CREATE TABLE IF NOT EXISTS diary_entries (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                entry_type TEXT DEFAULT 'reflection'
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_facts_person ON user_facts(person);
            CREATE INDEX IF NOT EXISTS idx_mood_ts ON mood_log(timestamp);
        """)
        self._conn.commit()

    # --- Conversations ---

    def start_conversation(self) -> str:
        conv_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO conversations (id, started_at) VALUES (?, ?)",
            (conv_id, _now()),
        )
        self._conn.commit()
        return conv_id

    def end_conversation(self, conv_id: str, summary: str | None = None) -> None:
        self._conn.execute(
            "UPDATE conversations SET ended_at=?, summary=? WHERE id=?",
            (_now(), summary, conv_id),
        )
        self._conn.commit()

    # --- Messages ---

    def add_message(self, conv_id: str, role: str, content: str) -> str:
        msg_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
            (msg_id, conv_id, role, content, _now()),
        )
        self._conn.commit()
        return msg_id

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE conversation_id=? ORDER BY timestamp",
            (conv_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- User Facts ---

    def add_fact(
        self,
        person: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source: str | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO user_facts (id, person, subject, predicate, object, confidence, source, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (fact_id, person, subject, predicate, obj, confidence, source, _now()),
        )
        self._conn.commit()
        return fact_id

    def get_facts_about(self, person: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM user_facts WHERE person=? AND superseded_by IS NULL ORDER BY created_at DESC",
            (person,),
        ).fetchall()
        return [dict(r) for r in rows]

    def supersede_fact(self, old_id: str, new_id: str) -> None:
        self._conn.execute(
            "UPDATE user_facts SET superseded_by=? WHERE id=?",
            (new_id, old_id),
        )
        self._conn.commit()

    # --- Mood Log ---

    def log_mood(self, emotion: str, intensity: float = 0.5, trigger: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO mood_log (id, timestamp, emotion, intensity, trigger) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), _now(), emotion, intensity, trigger),
        )
        self._conn.commit()

    def get_recent_moods(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT emotion, intensity, trigger, timestamp FROM mood_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Diary ---

    def add_diary_entry(self, content: str, entry_type: str = "reflection") -> str:
        entry_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO diary_entries (id, timestamp, content, entry_type) VALUES (?,?,?,?)",
            (entry_id, _now(), content, entry_type),
        )
        self._conn.commit()
        return entry_id

    def get_recent_diary(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT content, entry_type, timestamp FROM diary_entries ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Utilities ---

    def get_conversation_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()


def _now() -> str:
    return datetime.now().isoformat()
