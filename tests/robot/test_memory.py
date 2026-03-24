"""Tests for the memory system (SQLite + ChromaDB + MemoryGate)."""
import tempfile
from pathlib import Path

import pytest


def test_sqlite_store_conversations():
    from glados.robot.memory.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        conv_id = store.start_conversation()
        assert conv_id
        store.add_message(conv_id, "user", "hello")
        store.add_message(conv_id, "assistant", "hi there")
        msgs = store.get_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        store.end_conversation(conv_id, summary="test convo")
        store.close()


def test_sqlite_store_facts():
    from glados.robot.memory.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        fid = store.add_fact("creator", "creator", "имя", "Максим")
        facts = store.get_facts_about("creator")
        assert len(facts) == 1
        assert facts[0]["object"] == "Максим"

        # Supersede
        fid2 = store.add_fact("creator", "creator", "возраст", "25")
        store.supersede_fact(fid, fid2)
        facts = store.get_facts_about("creator")
        # Only non-superseded
        assert len(facts) == 1
        assert facts[0]["object"] == "25"
        store.close()


def test_sqlite_store_mood():
    from glados.robot.memory.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        store.log_mood("sarcastic", 0.8, "user said hello")
        moods = store.get_recent_moods(5)
        assert len(moods) == 1
        assert moods[0]["emotion"] == "sarcastic"
        store.close()


def test_sqlite_store_diary():
    from glados.robot.memory.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        store.add_diary_entry("Today was boring.", "reflection")
        entries = store.get_recent_diary(5)
        assert len(entries) == 1
        assert "boring" in entries[0]["content"]
        store.close()


def test_vector_store_add_and_search():
    from glados.robot.memory.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        vs = VectorStore(persist_dir=tmpdir)
        vs.add("User asked about philosophy of Descartes", importance=7)
        vs.add("User mentioned they like cats", importance=5)
        vs.add("User complained about weather", importance=3)

        results = vs.search("what do you think about philosophy", n_results=2)
        assert len(results) >= 1
        assert "philosophy" in results[0]["text"].lower() or "Descartes" in results[0]["text"]
        assert results[0]["score"] > 0


def test_vector_store_count():
    from glados.robot.memory.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        vs = VectorStore(persist_dir=tmpdir)
        assert vs.count() == 0
        vs.add("test memory")
        assert vs.count() == 1


def test_memory_gate_saves_important():
    from glados.robot.memory.memory_gate import MemoryGate
    from glados.robot.memory.sqlite_store import SQLiteStore
    from glados.robot.memory.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        sqlite = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        vector = VectorStore(persist_dir=str(Path(tmpdir) / "chroma"))
        gate = MemoryGate(sqlite=sqlite, vector=vector)

        # Personal info should be saved (high importance)
        gate.evaluate_and_save(
            user_text="Меня зовут Максим, мне 25 лет, я работаю программистом",
            assistant_text="Как интересно.",
            emotion="sarcastic",
            intensity=0.6,
            person="creator",
        )

        assert vector.count() >= 1
        facts = sqlite.get_facts_about("creator")
        assert len(facts) >= 1
        sqlite.close()


def test_memory_gate_skips_trivial():
    from glados.robot.memory.memory_gate import MemoryGate
    from glados.robot.memory.sqlite_store import SQLiteStore
    from glados.robot.memory.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        sqlite = SQLiteStore(db_path=Path(tmpdir) / "test.db")
        vector = VectorStore(persist_dir=str(Path(tmpdir) / "chroma"))
        gate = MemoryGate(sqlite=sqlite, vector=vector)

        # Short trivial exchange — should be skipped
        gate.evaluate_and_save(
            user_text="ок",
            assistant_text="Ок.",
            emotion="cold",
            intensity=0.2,
        )

        assert vector.count() == 0
        sqlite.close()
