import numpy as np
import pytest


def test_cosine_similarity():
    from glados.robot.face_id import cosine_similarity
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    from glados.robot.face_id import cosine_similarity
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_face_db_empty():
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    embedding = np.random.randn(512).astype(np.float32)
    matches = db.match(embedding, threshold=0.4)
    assert matches == []


def test_face_db_register_and_match():
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    db.register("alice", emb)
    matches = db.match(emb, threshold=0.4)
    assert len(matches) == 1
    assert matches[0][0] == "alice"
    assert matches[0][1] > 0.99


def test_face_db_no_false_match():
    from glados.robot.face_id import FaceDB
    db = FaceDB()
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0
    db.register("alice", emb_a)
    matches = db.match(emb_b, threshold=0.4)
    assert matches == []
