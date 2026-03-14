"""VisionWorker: camera → FastVLM + FaceID → VisionState + events."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..vision.fastvlm import FastVLM
from ..vision.vision_state import VisionState
from .config import VisionSettings
from .face_id import FaceRecognizer

if TYPE_CHECKING:
    from .face_display import FaceDisplay


@dataclass
class VisionEvent:
    description: str
    faces: list[str]
    change_score: float
    timestamp: float


class VisionWorker:
    """Camera → VLM description + face recognition → VisionState + events."""

    VISION_PROMPT = "Describe the image briefly, focusing on salient elements."

    def __init__(
        self,
        vision_state: VisionState,
        event_queue: queue.Queue,
        shutdown_event: threading.Event,
        settings: VisionSettings,
        face_recognizer: FaceRecognizer | None = None,
        vlm: FastVLM | None = None,
        face_display: "FaceDisplay | None" = None,
    ) -> None:
        self._state = vision_state
        self._event_q = event_queue
        self._shutdown = shutdown_event
        self._settings = settings
        self._face = face_recognizer
        self._vlm = vlm
        self._display = face_display

        self._capture: cv2.VideoCapture | None = None
        self._last_frame: NDArray[np.uint8] | None = None

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

                # Push every frame to FaceDisplay for PiP
                if self._display is not None:
                    self._display.update_camera(frame)

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
                    logger.success("Vision: {}", vision_text[:100])

                    # Push VLM text to display overlay
                    if self._display is not None:
                        self._display.update_vlm_text(vision_text)

                    try:
                        self._event_q.put_nowait(VisionEvent(
                            description=vision_text,
                            faces=face_names,
                            change_score=change,
                            timestamp=time.time(),
                        ))
                    except queue.Full:
                        pass

                if self._settings.save_frames:
                    self._save(frame, vision_text)

        except Exception as e:
            logger.exception("VisionWorker crash: {}", e)
        finally:
            if self._capture:
                self._capture.release()
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
