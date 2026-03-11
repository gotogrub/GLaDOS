"""VisionWorker: camera → FastVLM + FaceID → VisionState + events + cv2 display."""
from __future__ import annotations

import queue
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..vision.fastvlm import FastVLM
from ..vision.vision_state import VisionState
from .config import VisionSettings
from .face_id import FaceRecognizer

# Optional Pillow for Cyrillic text rendering
try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


@dataclass
class VisionEvent:
    description: str
    faces: list[str]
    change_score: float
    timestamp: float


class VisionWorker:
    """Camera → VLM description + face recognition → VisionState + events + OpenCV display."""

    VISION_PROMPT = "Describe the image briefly, focusing on salient elements."

    # Text panel settings
    _TEXT_PANEL_W = 480
    _TEXT_PANEL_H = 200
    _FONT_SIZE = 16
    _FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

    def __init__(
        self,
        vision_state: VisionState,
        event_queue: queue.Queue,
        shutdown_event: threading.Event,
        settings: VisionSettings,
        face_recognizer: FaceRecognizer | None = None,
        vlm: FastVLM | None = None,
        show_display: bool = True,
    ) -> None:
        self._state = vision_state
        self._event_q = event_queue
        self._shutdown = shutdown_event
        self._settings = settings
        self._face = face_recognizer
        self._vlm = vlm
        self._show = show_display

        self._capture: cv2.VideoCapture | None = None
        self._last_frame: NDArray[np.uint8] | None = None
        self._last_desc: str | None = None
        self._pil_font: ImageFont.FreeTypeFont | None = None

        if self._show and _PIL_AVAILABLE:
            try:
                self._pil_font = ImageFont.truetype(self._FONT_PATH, self._FONT_SIZE)
            except (OSError, IOError):
                logger.warning("Font {} not found, text panel disabled", self._FONT_PATH)

    def run(self) -> None:
        logger.info("VisionWorker started.")
        last_analysis = 0.0
        try:
            while not self._shutdown.is_set():
                if not self._ensure_camera():
                    self._shutdown.wait(timeout=1.0)
                    continue

                frame = self._grab()
                if frame is None:
                    self._shutdown.wait(timeout=0.01)
                    continue

                # Live preview — every frame, no delay
                if self._show:
                    cv2.imshow("GLaDOS Camera", frame)
                    cv2.waitKey(1)

                # Analysis — only at configured interval
                now = time.perf_counter()
                if now - last_analysis < self._settings.capture_interval_seconds:
                    continue
                last_analysis = now

                processed = self._resize(frame)
                change = self._scene_change(processed)

                if self._last_frame is not None and change <= self._settings.scene_change_threshold:
                    continue

                self._last_frame = processed.copy()

                # VLM description
                desc = ""
                if self._vlm:
                    try:
                        features = self._vlm.encode_image(frame)
                        desc = self._vlm.describe_from_features(
                            features, prompt=self.VISION_PROMPT,
                            max_tokens=self._settings.max_tokens,
                        ) or ""
                    except Exception as e:
                        logger.error("VLM failed: {}", e)

                # Face recognition
                face_results: list[dict] = []
                face_names: list[str] = []
                if self._face:
                    try:
                        face_results = self._face.recognize(frame)
                        face_names = [f["name"] for f in face_results if f["name"] != "unknown"]
                    except Exception as e:
                        logger.error("FaceID failed: {}", e)

                # Build combined vision text
                parts: list[str] = []
                if desc:
                    parts.append(desc)
                if face_names:
                    parts.append(f"Лица: {', '.join(face_names)}")
                elif face_results:
                    parts.append(f"Лица: {len(face_results)} незнакомых")
                vision_text = " ".join(parts)

                if vision_text:
                    self._state.update(vision_text)
                    self._last_desc = vision_text
                    logger.success("Vision: {}", vision_text[:100])

                    try:
                        self._event_q.put_nowait(VisionEvent(
                            description=vision_text,
                            faces=face_names,
                            change_score=change,
                            timestamp=time.time(),
                        ))
                    except queue.Full:
                        pass

                # Show snapshot with face bboxes + text panel
                if self._show:
                    self._show_snapshot(frame, face_results)
                    self._show_text_panel(vision_text)

                if self._settings.save_frames:
                    self._save(frame, vision_text)

        except Exception as e:
            logger.exception("VisionWorker crash: {}", e)
        finally:
            if self._capture:
                self._capture.release()
            if self._show:
                cv2.destroyAllWindows()
            logger.info("VisionWorker stopped.")

    def _ensure_camera(self) -> bool:
        if self._capture and self._capture.isOpened():
            return True
        if self._capture:
            self._capture.release()
        self._capture = cv2.VideoCapture(self._settings.camera_index)
        if not self._capture.isOpened():
            logger.error("Camera {} not available.", self._settings.camera_index)
            return False
        logger.success("Camera {} opened.", self._settings.camera_index)
        return True

    def _grab(self) -> NDArray[np.uint8] | None:
        assert self._capture is not None
        ret, frame = self._capture.read()
        return frame if ret and frame is not None else None

    def _resize(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w = frame.shape[:2]
        r = self._settings.resolution
        if max(h, w) <= r:
            return frame
        scale = r / max(h, w)
        return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _scene_change(self, frame: NDArray[np.uint8]) -> float:
        if self._last_frame is None:
            return 1.0
        if frame.shape != self._last_frame.shape:
            return 1.0
        diff = cv2.absdiff(frame, self._last_frame)
        return float(np.mean(diff)) / 255.0

    def _show_snapshot(self, frame: NDArray[np.uint8], faces: list[dict]) -> None:
        """Show analyzed frame with face bounding boxes."""
        display = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            sim = face.get("similarity", 0)
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            # ASCII-safe label for cv2.putText (names are folder names, ASCII)
            label = f"{name} ({sim:.2f})" if name != "unknown" else "?"
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("GLaDOS Snapshot", display)
        cv2.waitKey(1)

    def _show_text_panel(self, text: str) -> None:
        """Render VLM description as a text panel using PIL for Cyrillic support."""
        if not text:
            return
        if not _PIL_AVAILABLE or not self._pil_font:
            return

        img = Image.new("RGB", (self._TEXT_PANEL_W, self._TEXT_PANEL_H), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Word-wrap text to fit panel width
        wrapped = textwrap.fill(text, width=50)
        draw.text((10, 10), wrapped, fill=(200, 255, 200), font=self._pil_font)

        panel = np.array(img)
        panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imshow("GLaDOS VLM", panel)
        cv2.waitKey(1)

    def _save(self, frame: NDArray[np.uint8], desc: str) -> None:
        try:
            d = Path(self._settings.save_frames_dir)
            d.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(str(d / f"{ts}.jpg"), frame)
            if desc:
                (d / f"{ts}.txt").write_text(desc, encoding="utf-8")
            saved = sorted(d.glob("*.jpg"))
            if len(saved) > self._settings.save_frames_max:
                for old in saved[:len(saved) - self._settings.save_frames_max]:
                    old.unlink(missing_ok=True)
                    old.with_suffix(".txt").unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Frame save failed: {}", e)

