"""Fullscreen emotion face display for GLaDOS robot.

Shows emotion images on a dedicated monitor — acts as GLaDOS's "face".
During speech, alternates between speak frames for lip sync animation.
Uses pygame for reliable fullscreen rendering (OpenCV highgui is not thread-safe).

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

import os
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger


class FaceDisplay:
    """Displays emotion images in a fullscreen pygame window with lip sync.

    - Subscribes to "emotion" events to change expression.
    - Subscribes to "tts" / "tts_eos" events to toggle speaking animation.
    - When speaking: alternates speak_1 / speak_2 at configurable FPS.
    - When idle: shows the current emotion image.
    """

    WINDOW_TITLE = "GLaDOS Face"

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

    _EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

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
        self._req_width = width
        self._req_height = height

        self._current_emotion = default_emotion
        self._speaking = False
        self._speak_frame_idx = 0
        self._last_speak_switch = 0.0

        # Will be populated in _load_all (called from run loop thread)
        self._emotions: dict[str, Any] = {}  # emotion -> pygame.Surface
        self._speak_frames: list[Any] = []    # pygame.Surface list
        self._lock = threading.Lock()

        self._screen: Any = None  # pygame.Surface
        self._screen_w = 0
        self._screen_h = 0

    def _load_all(self) -> None:
        """Load emotion images and speak frames (must be called after pygame.init)."""
        import pygame

        loaded_files = 0

        for emotion, filename in self._EMOTION_TO_FILE.items():
            surf = self._load_file(filename, pygame)
            if surf is not None:
                self._emotions[emotion] = surf
                loaded_files += 1
            else:
                self._emotions[emotion] = self._generate_placeholder(emotion, pygame)

        for name in ("speak_1", "speak_2"):
            surf = self._load_file(name, pygame)
            if surf is not None:
                self._speak_frames.append(surf)
                loaded_files += 1

        logger.info(
            "FaceDisplay: {} emotions, {} speak frames ({} files loaded)",
            len(self._emotions),
            len(self._speak_frames),
            loaded_files,
        )

    def _find_file(self, name: str) -> Path | None:
        if not self._assets_dir.exists():
            return None
        for ext in self._EXTENSIONS:
            path = self._assets_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def _load_file(self, name: str, pygame: Any) -> Any | None:
        """Load image as pygame.Surface, scaled to screen size."""
        path = self._find_file(name)
        if path is None:
            return None
        try:
            surf = pygame.image.load(str(path)).convert()
            surf = pygame.transform.smoothscale(surf, (self._screen_w, self._screen_h))
            logger.debug("Loaded face asset: {} ({})", name, path.name)
            return surf
        except Exception as e:
            logger.warning("Failed to load {}: {}", path, e)
            return None

    def _generate_placeholder(self, emotion: str, pygame: Any) -> Any:
        """Generate a simple placeholder surface."""
        surf = pygame.Surface((self._screen_w, self._screen_h))
        surf.fill((0, 0, 0))
        font = pygame.font.SysFont("monospace", 48)
        text = font.render(emotion.upper(), True, (0, 180, 0))
        rect = text.get_rect(center=(self._screen_w // 2, self._screen_h // 2))
        surf.blit(text, rect)
        return surf

    # --- State setters (thread-safe) ---

    def set_emotion(self, emotion: str) -> None:
        emotion = emotion.lower()
        # Validate only after images are loaded (run() called).
        # Before that, accept any emotion — it will be validated at render time.
        if self._emotions and emotion not in self._emotions:
            logger.warning("Unknown emotion '{}', using default", emotion)
            emotion = self._default_emotion
        with self._lock:
            self._current_emotion = emotion
        logger.info("FaceDisplay: emotion → {}", emotion.upper())

    def set_speaking(self, speaking: bool) -> None:
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

    # --- Main display loop (runs in its own thread) ---

    def run(self, shutdown_event: threading.Event) -> None:
        """Main loop — call from a daemon thread."""
        import pygame

        # Ensure pygame uses the correct display
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

        pygame.init()

        # Get display info for fullscreen size
        if self._req_width > 0 and self._req_height > 0:
            self._screen_w = self._req_width
            self._screen_h = self._req_height
        else:
            info = pygame.display.Info()
            self._screen_w = info.current_w
            self._screen_h = info.current_h

        self._screen = pygame.display.set_mode(
            (self._screen_w, self._screen_h),
            pygame.FULLSCREEN | pygame.NOFRAME,
        )
        pygame.display.set_caption(self.WINDOW_TITLE)
        pygame.mouse.set_visible(False)

        logger.info(
            "FaceDisplay: pygame fullscreen {}x{} (monitor {})",
            self._screen_w, self._screen_h, self._monitor,
        )

        self._load_all()

        clock = pygame.time.Clock()

        while not shutdown_event.is_set():
            # Handle pygame events (required to keep window responsive)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    shutdown_event.set()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    shutdown_event.set()

            now = time.monotonic()

            with self._lock:
                speaking = self._speaking
                emotion = self._current_emotion

            if speaking and self._speak_frames:
                with self._lock:
                    if now - self._last_speak_switch >= self._SPEAK_FRAME_INTERVAL:
                        self._speak_frame_idx = (self._speak_frame_idx + 1) % len(self._speak_frames)
                        self._last_speak_switch = now
                    idx = self._speak_frame_idx
                surf = self._speak_frames[idx]
            else:
                surf = self._emotions.get(emotion)

            if surf is not None:
                self._screen.blit(surf, (0, 0))
                pygame.display.flip()

            clock.tick(30)  # 30 FPS

        pygame.quit()
        logger.info("FaceDisplay stopped.")
