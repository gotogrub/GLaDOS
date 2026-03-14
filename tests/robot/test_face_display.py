import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from glados.robot.face_display import FaceDisplay


def test_face_display_set_emotion():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, width=200, height=120)
        fd.set_emotion("anger")
        assert fd._current_emotion == "anger"


def test_face_display_unknown_emotion_before_load():
    """Before run() is called, any emotion is accepted (validated at render time)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, default_emotion="neutral", width=200, height=120)
        fd.set_emotion("nonexistent")
        # Before load, no validation — accepts whatever is set
        assert fd._current_emotion == "nonexistent"


def test_face_display_speaking_toggle():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, width=200, height=120)
        assert not fd._speaking
        fd.set_speaking(True)
        assert fd._speaking
        fd.set_speaking(False)
        assert not fd._speaking


def test_face_display_finds_image_files(tmp_path):
    # Create a test image
    (tmp_path / "normal.png").write_bytes(b"fake")
    fd = FaceDisplay(assets_dir=str(tmp_path))
    assert fd._find_file("normal") == tmp_path / "normal.png"
    assert fd._find_file("nonexistent") is None


def test_face_display_emotion_to_file_mapping():
    # Verify the mapping covers expected emotions
    fd = FaceDisplay(assets_dir="/nonexistent")
    assert "neutral" in fd._EMOTION_TO_FILE
    assert "anger" in fd._EMOTION_TO_FILE
    assert "sadness" in fd._EMOTION_TO_FILE
    assert "surprise" in fd._EMOTION_TO_FILE
    assert fd._EMOTION_TO_FILE["neutral"] == "normal"
    assert fd._EMOTION_TO_FILE["anger"] == "angry"
