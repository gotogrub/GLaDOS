"""InsightFace ONNX wrapper for face detection, embedding, and matching."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceDB:
    """In-memory face embedding database."""

    def __init__(self) -> None:
        self._embeddings: dict[str, NDArray[np.float32]] = {}

    def register(self, name: str, embedding: NDArray[np.float32]) -> None:
        self._embeddings[name] = embedding / np.linalg.norm(embedding)

    def match(
        self, embedding: NDArray[np.float32], threshold: float = 0.4
    ) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        query = embedding / np.linalg.norm(embedding)
        results = []
        for name, ref in self._embeddings.items():
            sim = cosine_similarity(query, ref)
            if sim >= threshold:
                results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @property
    def names(self) -> list[str]:
        return list(self._embeddings.keys())

    def __len__(self) -> int:
        return len(self._embeddings)


class FaceRecognizer:
    """Face detection + recognition using InsightFace ONNX models.

    Models required in model_dir:
    - det_scrfd_2.5g.onnx  (face detection)
    - arc_w600k_r18.onnx   (face embedding)
    """

    def __init__(self, model_dir: str | Path, face_db_dir: str | Path | None = None) -> None:
        import onnxruntime as ort

        self._model_dir = Path(model_dir)
        self._db = FaceDB()

        det_path = self._model_dir / "det_scrfd_2.5g.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"Face detection model not found: {det_path}")
        self._det_session = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"]
        )
        self._det_input_name = self._det_session.get_inputs()[0].name
        self._det_input_shape = self._det_session.get_inputs()[0].shape

        emb_path = self._model_dir / "arc_w600k_r18.onnx"
        if not emb_path.exists():
            raise FileNotFoundError(f"Face embedding model not found: {emb_path}")
        self._emb_session = ort.InferenceSession(
            str(emb_path), providers=["CPUExecutionProvider"]
        )
        self._emb_input_name = self._emb_session.get_inputs()[0].name

        if face_db_dir:
            self._load_face_db(Path(face_db_dir))

    def _load_face_db(self, face_db_dir: Path) -> None:
        if not face_db_dir.exists():
            logger.warning("Face DB directory not found: {}", face_db_dir)
            return
        for person_dir in sorted(face_db_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            embeddings = []
            for img_path in sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = self.detect(img)
                if faces:
                    largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                    emb = self.embed(img, largest)
                    if emb is not None:
                        embeddings.append(emb)
            if embeddings:
                avg = np.mean(embeddings, axis=0).astype(np.float32)
                self._db.register(name, avg)
                logger.success("FaceID: Loaded '{}' ({} photos)", name, len(embeddings))

    def detect(self, frame: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        target_size = (self._det_input_shape[3], self._det_input_shape[2])

        resized = cv2.resize(frame, target_size)
        blob = cv2.dnn.blobFromImage(
            resized, scalefactor=1.0 / 128.0, size=target_size,
            mean=(127.5, 127.5, 127.5), swapRB=True
        )

        outputs = self._det_session.run(None, {self._det_input_name: blob})

        boxes = []
        scale_x = w / target_size[0]
        scale_y = h / target_size[1]

        for i in range(0, len(outputs), 3):
            if i + 1 >= len(outputs):
                break
            scores = outputs[i]
            bboxes = outputs[i + 1]

            for j in range(scores.shape[0]):
                for k in range(scores.shape[1]):
                    score = float(scores[j, k, 0]) if scores.ndim == 3 else float(scores[j, k])
                    if score < 0.5:
                        continue
                    if bboxes.ndim == 3:
                        bbox = bboxes[j, k]
                    else:
                        bbox = bboxes[j]
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    boxes.append((x1, y1, x2, y2))

        return boxes

    def embed(
        self, frame: NDArray[np.uint8], bbox: tuple[int, int, int, int]
    ) -> NDArray[np.float32] | None:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 127.5
        face_input = face_norm.transpose(2, 0, 1)[np.newaxis]

        try:
            result = self._emb_session.run(None, {self._emb_input_name: face_input})
            return result[0][0].astype(np.float32)
        except Exception as e:
            logger.warning("FaceID embed failed: {}", e)
            return None

    def recognize(
        self, frame: NDArray[np.uint8], threshold: float = 0.4
    ) -> list[dict]:
        faces = self.detect(frame)
        results = []
        for bbox in faces:
            emb = self.embed(frame, bbox)
            if emb is None:
                results.append({"name": "unknown", "similarity": 0.0, "bbox": bbox})
                continue
            matches = self._db.match(emb, threshold=threshold)
            if matches:
                name, sim = matches[0]
                results.append({"name": name, "similarity": sim, "bbox": bbox})
            else:
                results.append({"name": "unknown", "similarity": 0.0, "bbox": bbox})
        return results

    @property
    def db(self) -> FaceDB:
        return self._db
