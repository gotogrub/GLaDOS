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
    """Face detection + recognition using InsightFace ONNX models (buffalo_s).

    Models required in model_dir:
    - det_500m.onnx      (SCRFD face detection, ~2.5MB)
    - w600k_mbf.onnx     (MobileFaceNet embedding, ~13MB)
    """

    _DET_SIZE: int = 640
    _STRIDES: tuple[int, ...] = (8, 16, 32)
    _SCORE_THRESH: float = 0.65
    _NMS_THRESH: float = 0.4

    def __init__(self, model_dir: str | Path, face_db_dir: str | Path | None = None) -> None:
        import onnxruntime as ort

        self._model_dir = Path(model_dir)
        self._db = FaceDB()

        det_path = self._model_dir / "det_500m.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"Face detection model not found: {det_path}")
        self._det_session = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"]
        )
        self._det_input_name = self._det_session.get_inputs()[0].name

        emb_path = self._model_dir / "w600k_mbf.onnx"
        if not emb_path.exists():
            raise FileNotFoundError(f"Face embedding model not found: {emb_path}")
        self._emb_session = ort.InferenceSession(
            str(emb_path), providers=["CPUExecutionProvider"]
        )
        self._emb_input_name = self._emb_session.get_inputs()[0].name

        # Pre-build anchor grids for each stride
        self._anchors = self._build_anchors()

        if face_db_dir:
            self._load_face_db(Path(face_db_dir))

    def _build_anchors(self) -> dict[int, NDArray[np.float32]]:
        """Pre-compute anchor centers for each stride."""
        anchors = {}
        for stride in self._STRIDES:
            feat = self._DET_SIZE // stride
            grid_y, grid_x = np.mgrid[:feat, :feat].astype(np.float32)
            centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1) * stride
            # 2 anchors per position
            centers = np.concatenate([centers, centers], axis=0)
            anchors[stride] = centers
        return anchors

    def _load_face_db(self, face_db_dir: Path) -> None:
        if not face_db_dir.exists():
            logger.warning("Face DB directory not found: {}", face_db_dir)
            return
        # Use a softer detection threshold for DB enrollment photos
        # (photos may be small, cropped, or lower quality than live camera)
        saved_thresh = self._SCORE_THRESH
        self._SCORE_THRESH = 0.3
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
                else:
                    logger.warning("FaceID: No face in {}", img_path.name)
            if embeddings:
                avg = np.mean(embeddings, axis=0).astype(np.float32)
                self._db.register(name, avg)
                logger.success("FaceID: Loaded '{}' ({} photos)", name, len(embeddings))
            else:
                logger.warning("FaceID: No usable faces for '{}'", name)
        self._SCORE_THRESH = saved_thresh

    def detect(self, frame: NDArray[np.uint8]) -> list[tuple[int, int, int, int]]:
        """Detect faces using SCRFD. Returns list of (x1, y1, x2, y2) in original frame coords."""
        h, w = frame.shape[:2]
        sz = self._DET_SIZE

        # Preprocess: resize with letterbox, normalize
        scale = min(sz / h, sz / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (nw, nh))
        padded = np.full((sz, sz, 3), 0, dtype=np.uint8)
        padded[:nh, :nw] = resized

        blob = (padded.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

        outputs = self._det_session.run(None, {self._det_input_name: blob})
        # Outputs: [scores_8, scores_16, scores_32, bboxes_8, bboxes_16, bboxes_32, kps_8, kps_16, kps_32]
        n_strides = len(self._STRIDES)
        all_scores = outputs[:n_strides]
        all_bboxes = outputs[n_strides:n_strides * 2]

        boxes = []
        scores = []
        for i, stride in enumerate(self._STRIDES):
            sc = all_scores[i].ravel()  # (N,)
            bb = all_bboxes[i]  # (N, 4) — distances from anchor
            mask = sc > self._SCORE_THRESH
            if not mask.any():
                continue
            sc = sc[mask]
            bb = bb[mask]
            anchors = self._anchors[stride][mask]

            # Decode: anchor_center ± distance * stride
            x1 = anchors[:, 0] - bb[:, 0] * stride
            y1 = anchors[:, 1] - bb[:, 1] * stride
            x2 = anchors[:, 0] + bb[:, 2] * stride
            y2 = anchors[:, 1] + bb[:, 3] * stride

            for j in range(len(sc)):
                boxes.append([x1[j], y1[j], x2[j], y2[j]])
                scores.append(float(sc[j]))

        if not boxes:
            return []

        # NMS
        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_arr.tolist(), scores_arr.tolist(),
            self._SCORE_THRESH, self._NMS_THRESH,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        # Map back to original frame coordinates
        result = []
        for idx in indices:
            bx = boxes_arr[idx]
            ox1 = int(bx[0] / scale)
            oy1 = int(bx[1] / scale)
            ox2 = int(bx[2] / scale)
            oy2 = int(bx[3] / scale)
            result.append((
                max(0, ox1), max(0, oy1),
                min(w, ox2), min(h, oy2),
            ))
        return result

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
        self, frame: NDArray[np.uint8], threshold: float = 0.6
    ) -> list[dict]:
        faces = self.detect(frame)
        results = []
        for bbox in faces:
            # Skip tiny detections (likely false positives)
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if bw < 40 or bh < 40:
                continue
            emb = self.embed(frame, bbox)
            if emb is None:
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
