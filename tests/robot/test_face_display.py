import tempfile
from pathlib import Path

import cv2
import numpy as np

from glados.robot.face_display import FaceDisplay


def test_face_display_generates_placeholders():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, width=200, height=120)
        assert "neutral" in fd._emotions
        assert "anger" in fd._emotions
        assert fd._emotions["neutral"].shape == (120, 200, 3)


def test_face_display_set_emotion():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, width=200, height=120)
        fd.set_emotion("anger")
        assert fd._current_emotion == "anger"


def test_face_display_unknown_emotion_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, default_emotion="neutral", width=200, height=120)
        fd.set_emotion("nonexistent")
        assert fd._current_emotion == "neutral"


def test_face_display_loads_image_from_file(tmp_path):
    # Create a test image named "normal.png" (maps to "neutral" emotion)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    cv2.imwrite(str(tmp_path / "normal.png"), img)

    fd = FaceDisplay(assets_dir=str(tmp_path), width=200, height=120)
    # neutral maps to "normal" file
    assert fd._emotions["neutral"].shape == (120, 200, 3)


def test_face_display_loads_speak_frames(tmp_path):
    for name in ("speak_1", "speak_2"):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{name}.png"), img)

    fd = FaceDisplay(assets_dir=str(tmp_path), width=200, height=120)
    assert len(fd._speak_frames) == 2


def test_face_display_speaking_toggle():
    with tempfile.TemporaryDirectory() as tmpdir:
        fd = FaceDisplay(assets_dir=tmpdir, width=200, height=120)
        assert not fd._speaking
        fd.set_speaking(True)
        assert fd._speaking
        fd.set_speaking(False)
        assert not fd._speaking


def test_face_display_native_resolution(tmp_path):
    # width=0, height=0 → use original image size
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "normal.png"), img)

    fd = FaceDisplay(assets_dir=str(tmp_path))
    assert fd._emotions["neutral"].shape == (480, 640, 3)
