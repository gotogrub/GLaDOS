"""Fullscreen emotion face display for GLaDOS robot.

Shows emotion images on a dedicated monitor — acts as GLaDOS's "face".
During speech, alternates between speak frames for lip sync animation.

Assets directory layout:
    assets/faces/
        normal.png      — neutral / default
        angry.png       — anger
        sad.png         — sadness
        surprised.png   — surprise
        speak_1.png     — mouth open frame 1
        speak_2.png     — mouth open frame 2
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger


class FaceDisplay:
    """Displays emotion images in a fullscreen OpenCV window with lip sync.

    - Subscribes to "emotion" events to change expression.
    - Subscribes to "tts" / "tts_eos" events to toggle speaking animation.
    - When speaking: alternates speak_1 / speak_2 at ~8 FPS.
    - When idle: shows the current emotion image.
    """

    WINDOW_NAME = "GLaDOS Face"
    _EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    # Map emotion tags → asset filenames (without extension)
    _EMOTION_TO_FILE = {
        "neutral":   "normal",
        "sarcasm":   "normal",
        "anger":     "angry",
        "curiosity": "normal",
        "disgust":   "angry",
        "amusement": "normal",
        "sadness":   "sad",
        "surprise":  "surprised",
    }

    # Speak animation speed (seconds per frame)
    _SPEAK_FRAME_INTERVAL = 0.12  # ~8 FPS

    def __init__(
        self,
        assets_dir: str | Path = "assets/faces",
        default_emotion: str = "neutral",
        monitor: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        self._assets_dir = Path(assets_dir)
        self._default_emotion = default_emotion
        self._monitor = monitor
        self._width = width
        self._height = height

        self._current_emotion = default_emotion
        self._speaking = False
        self._speak_frame_idx = 0
        self._last_speak_switch = 0.0

        self._emotions: dict[str, np.ndarray] = {}
        self._speak_frames: list[np.ndarray] = []
        self._lock = threading.Lock()

        self._load_all()

    def _load_all(self) -> None:
        """Load emotion images and speak frames from assets."""
        loaded_files = 0

        # Load emotion images
        for emotion, filename in self._EMOTION_TO_FILE.items():
            img = self._load_file(filename)
            if img is not None:
                self._emotions[emotion] = img
                loaded_files += 1
            else:
                self._emotions[emotion] = self._generate_placeholder(emotion)

        # De-duplicate: emotions sharing the same file get the same array
        # (already handled by loading from filename)

        # Load speak frames
        for name in ("speak_1", "speak_2"):
            img = self._load_file(name)
            if img is not None:
                self._speak_frames.append(img)
                loaded_files += 1

        logger.info(
            "FaceDisplay: {} emotions, {} speak frames ({} files loaded)",
            len(self._emotions),
            len(self._speak_frames),
            loaded_files,
        )

    def _find_file(self, name: str) -> Path | None:
        """Find image file by name (without extension)."""
        if not self._assets_dir.exists():
            return None
        for ext in self._EXTENSIONS:
            path = self._assets_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _load_file(self, name: str) -> np.ndarray | None:
        """Load and optionally resize an image."""
        path = self._find_file(name)
        if path is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read image: {}", path)
            return None
        if self._width > 0 and self._height > 0:
            img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        logger.debug("Loaded face asset: {} ({})", name, path.name)
        return img

    def _generate_placeholder(self, emotion: str) -> np.ndarray:
        """Generate a simple placeholder when no image file exists."""
        w = self._width or 800
        h = self._height or 480
        img = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = emotion.upper()
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        x = (w - text_size[0]) // 2
        y = (h + text_size[1]) // 2
        cv2.putText(img, text, (x, y), font, 1.0, (0, 180, 0), 2, cv2.LINE_AA)
        return img

    # --- State setters (thread-safe) ---

    def set_emotion(self, emotion: str) -> None:
        """Change the displayed emotion."""
        emotion = emotion.lower()
        if emotion not in self._emotions:
            logger.warning("Unknown emotion '{}', using default", emotion)
            emotion = self._default_emotion
        with self._lock:
            self._current_emotion = emotion
        logger.info("FaceDisplay: emotion → {}", emotion.upper())

    def set_speaking(self, speaking: bool) -> None:
        """Toggle speaking animation."""
        with self._lock:
            self._speaking = speaking
            if speaking:
                self._speak_frame_idx = 0
                self._last_speak_switch = time.monotonic()

    # --- EventBus handlers ---

    async def handle_emotion_event(self, event: Any) -> None:
        emotion = event.data.get("emotion", self._default_emotion)
        self.set_emotion(emotion)

    async def handle_tts_event(self, event: Any) -> None:
        self.set_speaking(True)

    async def handle_tts_eos_event(self, event: Any) -> None:
        self.set_speaking(False)

    # --- Display ---

    def show(self) -> None:
        """Display the appropriate frame. Call from the display thread."""
        now = time.monotonic()

        with self._lock:
            speaking = self._speaking
            emotion = self._current_emotion

        if speaking and self._speak_frames:
            # Alternate speak frames for lip sync
            with self._lock:
                if now - self._last_speak_switch >= self._SPEAK_FRAME_INTERVAL:
                    self._speak_frame_idx = (self._speak_frame_idx + 1) % len(self._speak_frames)
                    self._last_speak_switch = now
                idx = self._speak_frame_idx
            img = self._speak_frames[idx]
        else:
            img = self._emotions.get(emotion)

        if img is not None:
            cv2.imshow(self.WINDOW_NAME, img)

    def setup_window(self) -> None:
        """Create and configure the fullscreen window."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        logger.info("FaceDisplay: fullscreen window created (monitor {})", self._monitor)
        self.show()
        cv2.waitKey(1)

    def destroy(self) -> None:
        """Close the display window."""
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass
