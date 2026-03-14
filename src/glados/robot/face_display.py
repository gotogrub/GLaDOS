"""Unified GLaDOS display — emotion face, camera PiP, VLM text overlay.

Single pygame window combining:
- Emotion face (fullscreen background)
- Lip sync animation (speak_1/speak_2 frames during audio playback)
- Camera preview (picture-in-picture, bottom-right corner)
- VLM description text (bottom overlay)

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
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class FaceDisplay:
    """Unified pygame display for GLaDOS robot.

    Thread-safe: VisionWorker pushes camera frames and VLM text via
    update_camera() and update_vlm_text(). FaceDisplay renders everything
    in its own thread.
    """

    WINDOW_TITLE = "GLaDOS"

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

    _SPEAK_FRAME_INTERVAL = 0.12  # ~8 FPS lip sync
    _EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    # Camera PiP settings (fraction of screen)
    _PIP_SCALE = 0.2  # 20% of screen width
    _PIP_MARGIN = 10  # pixels from edge

    # VLM text overlay
    _TEXT_PADDING = 10
    _TEXT_FONT_SIZE = 20
    _TEXT_BG_ALPHA = 180  # semi-transparent background

    def __init__(
        self,
        assets_dir: str | Path = "assets/faces",
        default_emotion: str = "neutral",
        monitor: int = 0,
        width: int = 0,
        height: int = 0,
        speaking_event: threading.Event | None = None,
    ) -> None:
        self._assets_dir = Path(assets_dir)
        self._default_emotion = default_emotion
        self._monitor = monitor
        self._req_width = width
        self._req_height = height

        self._current_emotion = default_emotion
        self._speaking_event = speaking_event
        self._speak_frame_idx = 0
        self._last_speak_switch = 0.0

        self._emotions: dict[str, Any] = {}
        self._speak_frames: list[Any] = []
        self._lock = threading.Lock()

        # Shared state from VisionWorker (thread-safe)
        self._camera_frame: np.ndarray | None = None
        self._vlm_text: str = ""

        self._screen: Any = None
        self._screen_w = 0
        self._screen_h = 0
        self._font: Any = None

    # --- Thread-safe state updates ---

    def set_emotion(self, emotion: str) -> None:
        emotion = emotion.lower()
        if self._emotions and emotion not in self._emotions:
            logger.warning("Unknown emotion '{}', using default", emotion)
            emotion = self._default_emotion
        with self._lock:
            self._current_emotion = emotion
        logger.info("FaceDisplay: emotion → {}", emotion.upper())

    def update_camera(self, frame: np.ndarray) -> None:
        """Push a new camera frame (called from VisionWorker thread)."""
        with self._lock:
            self._camera_frame = frame

    def update_vlm_text(self, text: str) -> None:
        """Push new VLM description (called from VisionWorker thread)."""
        with self._lock:
            self._vlm_text = text

    # --- EventBus handler ---

    async def handle_emotion_event(self, event: Any) -> None:
        emotion = event.data.get("emotion", self._default_emotion)
        self.set_emotion(emotion)

    # --- Asset loading ---

    def _load_all(self) -> None:
        import pygame

        loaded = 0
        for emotion, filename in self._EMOTION_TO_FILE.items():
            surf = self._load_file(filename, pygame)
            if surf is not None:
                self._emotions[emotion] = surf
                loaded += 1
            else:
                self._emotions[emotion] = self._generate_placeholder(emotion, pygame)

        for name in ("speak_1", "speak_2"):
            surf = self._load_file(name, pygame)
            if surf is not None:
                self._speak_frames.append(surf)
                loaded += 1

        logger.info(
            "FaceDisplay: {} emotions, {} speak frames ({} files loaded)",
            len(self._emotions), len(self._speak_frames), loaded,
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
        surf = pygame.Surface((self._screen_w, self._screen_h))
        surf.fill((0, 0, 0))
        font = pygame.font.SysFont("monospace", 48)
        text = font.render(emotion.upper(), True, (0, 180, 0))
        rect = text.get_rect(center=(self._screen_w // 2, self._screen_h // 2))
        surf.blit(text, rect)
        return surf

    # --- Rendering helpers ---

    def _render_pip(self, pygame: Any) -> None:
        """Render camera picture-in-picture in bottom-right corner."""
        with self._lock:
            frame = self._camera_frame

        if frame is None:
            return

        # Convert OpenCV BGR → RGB and create pygame surface
        h, w = frame.shape[:2]
        pip_w = int(self._screen_w * self._PIP_SCALE)
        pip_h = int(pip_w * h / w) if w > 0 else pip_w

        try:
            # OpenCV frame is BGR, pygame wants RGB
            rgb = frame[:, :, ::-1].copy()
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            surf = pygame.transform.smoothscale(surf, (pip_w, pip_h))

            # Position: bottom-right with margin
            x = self._screen_w - pip_w - self._PIP_MARGIN
            y = self._screen_h - pip_h - self._PIP_MARGIN

            # Border
            border_rect = pygame.Rect(x - 2, y - 2, pip_w + 4, pip_h + 4)
            pygame.draw.rect(self._screen, (0, 180, 0), border_rect, 2)

            self._screen.blit(surf, (x, y))
        except Exception:
            pass  # skip frame on error

    def _render_text(self, pygame: Any) -> None:
        """Render VLM text overlay at bottom of screen."""
        with self._lock:
            text = self._vlm_text

        if not text or not self._font:
            return

        wrapped = textwrap.fill(text, width=60)
        lines = wrapped.split("\n")

        line_h = self._font.get_linesize()
        total_h = line_h * len(lines) + self._TEXT_PADDING * 2

        # Semi-transparent background
        bg = pygame.Surface((self._screen_w, total_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, self._TEXT_BG_ALPHA))
        self._screen.blit(bg, (0, self._screen_h - total_h))

        # Text lines
        y = self._screen_h - total_h + self._TEXT_PADDING
        for line in lines:
            rendered = self._font.render(line, True, (0, 220, 0))
            self._screen.blit(rendered, (self._TEXT_PADDING, y))
            y += line_h

    # --- Main loop ---

    def run(self, shutdown_event: threading.Event) -> None:
        """Main display loop — call from a daemon thread."""
        import pygame

        # Use windowed mode (not fullscreen) so user can move/resize
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

        pygame.init()

        # Screen size
        if self._req_width > 0 and self._req_height > 0:
            self._screen_w = self._req_width
            self._screen_h = self._req_height
        else:
            info = pygame.display.Info()
            self._screen_w = info.current_w
            self._screen_h = info.current_h

        self._screen = pygame.display.set_mode(
            (self._screen_w, self._screen_h),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(self.WINDOW_TITLE)

        # Font for VLM text overlay
        try:
            self._font = pygame.font.Font(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                self._TEXT_FONT_SIZE,
            )
        except Exception:
            self._font = pygame.font.SysFont("monospace", self._TEXT_FONT_SIZE)

        self._load_all()

        logger.info(
            "FaceDisplay: {}x{} window (monitor {})",
            self._screen_w, self._screen_h, self._monitor,
        )

        clock = pygame.time.Clock()
        fullscreen = False

        while not shutdown_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    shutdown_event.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        shutdown_event.set()
                    elif event.key == pygame.K_F11 or event.key == pygame.K_f:
                        # Toggle fullscreen with F11 or F
                        fullscreen = not fullscreen
                        if fullscreen:
                            self._screen = pygame.display.set_mode(
                                (self._screen_w, self._screen_h),
                                pygame.FULLSCREEN | pygame.NOFRAME,
                            )
                        else:
                            self._screen = pygame.display.set_mode(
                                (self._screen_w, self._screen_h),
                                pygame.RESIZABLE,
                            )
                elif event.type == pygame.VIDEORESIZE:
                    self._screen_w = event.w
                    self._screen_h = event.h
                    self._screen = pygame.display.set_mode(
                        (self._screen_w, self._screen_h),
                        pygame.RESIZABLE,
                    )
                    # Reload assets at new size
                    self._emotions.clear()
                    self._speak_frames.clear()
                    self._load_all()

            now = time.monotonic()

            # Determine face to show
            speaking = (
                self._speaking_event is not None
                and self._speaking_event.is_set()
            )

            with self._lock:
                emotion = self._current_emotion

            if speaking and self._speak_frames:
                if now - self._last_speak_switch >= self._SPEAK_FRAME_INTERVAL:
                    self._speak_frame_idx = (self._speak_frame_idx + 1) % len(self._speak_frames)
                    self._last_speak_switch = now
                face_surf = self._speak_frames[self._speak_frame_idx]
            else:
                face_surf = self._emotions.get(emotion)

            # Draw: face background → camera PiP → text overlay
            if face_surf is not None:
                self._screen.blit(face_surf, (0, 0))
            else:
                self._screen.fill((0, 0, 0))

            self._render_pip(pygame)
            self._render_text(pygame)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        logger.info("FaceDisplay stopped.")
