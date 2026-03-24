"""ChromaDB vector store for semantic memory search."""
from __future__ import annotations

import math
import time
from typing import Any

from loguru import logger


class VectorStore:
    """Semantic memory backed by ChromaDB.

    Stores episodic memories (conversation snippets, events) with embeddings
    for similarity search. Retrieval scoring uses the Generative Agents formula:
        score = α1 * recency + α2 * (importance/10) + α3 * similarity
    """

    _COLLECTION_NAME = "episodic_memory"
    # Scoring weights (from Generative Agents paper)
    _ALPHA_RECENCY = 1.0
    _ALPHA_IMPORTANCE = 1.0
    _ALPHA_SIMILARITY = 1.0
    # Recency decay: half-life in hours
    _RECENCY_HALF_LIFE_HOURS = 24.0

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore opened: {} ({} memories)",
            persist_dir,
            self._collection.count(),
        )

    def add(
        self,
        text: str,
        importance: int = 5,
        participants: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a memory. Returns its ID."""
        import uuid

        mem_id = str(uuid.uuid4())
        meta = {
            "importance": importance,
            "participants": ",".join(participants or []),
            "timestamp": time.time(),
            "access_count": 0,
            **(metadata or {}),
        }
        self._collection.add(
            ids=[mem_id],
            documents=[text],
            metadatas=[meta],
        )
        logger.debug("Memory added: '{}' (importance={})", text[:60], importance)
        return mem_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        person_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories by semantic similarity with Generative Agents scoring.

        Returns list of dicts with: text, score, importance, timestamp, participants.
        """
        where = None
        if person_filter:
            where = {"participants": {"$contains": person_filter}}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results * 3, max(20, n_results)),  # over-fetch for re-ranking
                where=where,
            )
        except Exception as e:
            logger.warning("VectorStore search failed: {}", e)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        now = time.time()
        scored: list[dict[str, Any]] = []

        for i, mem_id in enumerate(results["ids"][0]):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0.5
            similarity = 1.0 - distance  # cosine distance → similarity

            ts = meta.get("timestamp", now)
            hours_ago = (now - ts) / 3600.0
            recency = math.exp(-0.693 * hours_ago / self._RECENCY_HALF_LIFE_HOURS)

            importance = meta.get("importance", 5)
            score = (
                self._ALPHA_RECENCY * recency
                + self._ALPHA_IMPORTANCE * (importance / 10.0)
                + self._ALPHA_SIMILARITY * similarity
            )

            scored.append({
                "id": mem_id,
                "text": doc,
                "score": score,
                "similarity": similarity,
                "recency": recency,
                "importance": importance,
                "timestamp": ts,
                "participants": meta.get("participants", ""),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n_results]

    def count(self) -> int:
        return self._collection.count()

    def delete(self, mem_id: str) -> None:
        self._collection.delete(ids=[mem_id])
