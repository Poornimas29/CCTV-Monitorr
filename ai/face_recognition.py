"""InsightFace-based face recognition for registered employees.

This module replaces the previous custom matcher with a more robust backend
that uses InsightFace when available for face detection, alignment, and
embedding generation. It preserves the existing project interfaces so the
streaming and dashboard modules continue to work unchanged.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    """Loads employee reference images and recognizes faces in camera frames."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        cache_path: Optional[str] = None,
        detector_path: Optional[str] = None,
        threshold: float = 0.40,
        debug: bool = False,
    ) -> None:
        self._project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self._cache_path = Path(cache_path or self._project_root / ".cache" / "face_embeddings.json").resolve()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._employee_embeddings: Dict[str, Dict[str, Any]] = {}
        self._employee_lookup: Dict[str, Dict[str, Any]] = {}
        self._detector = None
        self._insightface_app = None
        self._backend_name = "custom"

        # Load threshold from environment or parameter
        env_threshold = os.getenv("FACE_RECOGNITION_THRESHOLD")
        if env_threshold is not None:
            try:
                threshold = float(env_threshold)
            except ValueError:
                pass
        self._threshold = threshold
        self._debug = debug

        if detector_path:
            self._detector = cv2.CascadeClassifier(detector_path)
        else:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                self._detector = cv2.CascadeClassifier(str(cascade_path))

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._insightface_app = app
            self._backend_name = "insightface"
            logger.info("InsightFace backend loaded successfully")
        except Exception as exc:
            logger.warning("InsightFace backend unavailable; using OpenCV fallback: %s", exc)
            self._backend_name = "custom"

    def initialize(self, employee_manager: Any) -> Dict[str, Any]:
        """Load employee metadata, create or load embeddings, and cache results."""
        employees = employee_manager.get_all_employees()
        self._employee_lookup = {emp["employee_id"]: emp for emp in employees}

        cache_payload: Dict[str, Any] = self._load_cache()
        self._employee_embeddings = {}

        for employee in employees:
            emp_id = employee["employee_id"]
            image_paths = employee_manager.get_employee_images(emp_id)
            cache_entry = cache_payload.get(emp_id)

            # Build current images metadata
            current_metadata = []
            for path_str in image_paths:
                p = Path(path_str)
                if p.exists():
                    stat = p.stat()
                    current_metadata.append({
                        "path": p.name,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size
                    })

            expected_dim = 512 if self._backend_name == "insightface" else 4096
            cached_metadata = cache_entry.get("images_metadata") if cache_entry else None
            is_cache_valid = False

            if cache_entry and cached_metadata and len(cached_metadata) == len(current_metadata):
                matches = True
                for c_meta, r_meta in zip(current_metadata, cached_metadata):
                    if (c_meta["path"] != r_meta.get("path") or 
                        abs(c_meta["mtime"] - r_meta.get("mtime", 0.0)) > 0.1 or 
                        c_meta["size"] != r_meta.get("size")):
                        matches = False
                        break
                if matches:
                    embedding = np.asarray(cache_entry.get("embedding", []), dtype=np.float32)
                    if embedding.size == expected_dim:
                        is_cache_valid = True
                        self._employee_embeddings[emp_id] = {
                            "employee_id": emp_id,
                            "name": employee.get("name", emp_id),
                            "embedding": embedding,
                            "image_count": len(image_paths),
                            "images_metadata": current_metadata,
                        }

            if is_cache_valid:
                continue

            embedding = self._build_employee_embedding(image_paths)
            if embedding is None:
                continue

            self._employee_embeddings[emp_id] = {
                "employee_id": emp_id,
                "name": employee.get("name", emp_id),
                "embedding": embedding,
                "image_count": len(image_paths),
                "images_metadata": current_metadata,
            }

        self._save_cache()
        return {
            "registered_employees": len(self._employee_embeddings),
            "total_face_images": sum(
                len(employee_manager.get_employee_images(emp_id)) for emp_id in self._employee_embeddings
            ),
            "embeddings_loaded": bool(self._employee_embeddings),
        }

    def recognize_frame(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Identify the best-matching employee only when the similarity exceeds the threshold."""
        if frame is None or frame.size == 0:
            self._debug_print("Frame Received: No\nNumber of Faces Detected: 0\nBest Match: None\nRecognition Threshold: {threshold}\nFinal Decision: Unknown")
            return {**self._unknown_result(), "status": "no_face"}

        self._debug_print(f"Frame Received: Yes\nNumber of Faces Detected: evaluating")
        detections = self._detect_faces(frame)
        if not detections:
            self._debug_print(
                f"Frame Received: Yes\nNumber of Faces Detected: 0\nBest Match: None\nRecognition Threshold: {self._threshold:.2f}\nFinal Decision: Unknown"
            )
            return {**self._unknown_result(), "bbox": None, "status": "no_face"}

        best_employee_id: Optional[str] = None
        best_name: str = "Unknown"
        best_similarity = 0.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_embedding: Optional[np.ndarray] = None
        similarity_details: List[str] = []

        for detection in detections:
            embedding = detection.get("embedding")
            if embedding is None:
                embedding = self._build_embedding(detection.get("face"))
            if embedding is None or not self._employee_embeddings:
                continue

            for emp_id, record in self._employee_embeddings.items():
                similarity = self._cosine_similarity(embedding, record["embedding"])
                similarity_details.append(f"{emp_id}: {similarity:.4f}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_employee_id = emp_id
                    best_name = record.get("name", emp_id)
                    best_bbox = detection.get("bbox")
                    best_embedding = embedding

        matched = best_employee_id is not None and best_similarity >= self._threshold
        decision = "Recognized" if matched else "Unknown"
        debug_lines = [
            "Frame Received: Yes",
            f"Number of Faces Detected: {len(detections)}",
        ]
        if similarity_details:
            debug_lines.extend(similarity_details)
        debug_lines.extend(
            [
                f"Best Match: {best_employee_id or 'None'}",
                f"Recognition Threshold: {self._threshold:.2f}",
                f"Final Decision: {decision}",
            ]
        )
        self._debug_print("\n".join(debug_lines))

        return {
            "employee_id": best_employee_id if matched else None,
            "best_employee_id": best_employee_id,
            "employee_name": best_name if matched else "Unknown",
            "confidence": round(float(best_similarity * 100.0), 1) if matched else 0.0,
            "matched": matched,
            "bbox": best_bbox,
            "embedding": best_embedding,
            "status": "recognized" if matched else "unknown",
            "similarity": round(float(best_similarity), 3),
        }

    def recognize_crop(self, crop: Optional[np.ndarray]) -> Dict[str, Any]:
        """Recognize a face within a cropped person image."""
        if crop is None or crop.size == 0:
            return self._unknown_result()

        detections = self._detect_faces(crop)
        if not detections:
            return {**self._unknown_result(), "status": "no_face"}

        best_employee_id: Optional[str] = None
        best_name: str = "Unknown"
        best_similarity = 0.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_embedding: Optional[np.ndarray] = None

        for detection in detections:
            embedding = detection.get("embedding")
            if embedding is None:
                embedding = self._build_embedding(detection.get("face"))
            if embedding is None or not self._employee_embeddings:
                continue

            for emp_id, record in self._employee_embeddings.items():
                similarity = self._cosine_similarity(embedding, record["embedding"])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_employee_id = emp_id
                    best_name = record.get("name", emp_id)
                    best_bbox = detection.get("bbox")
                    best_embedding = embedding

        matched = best_employee_id is not None and best_similarity >= self._threshold
        return {
            "employee_id": best_employee_id if matched else None,
            "best_employee_id": best_employee_id,
            "employee_name": best_name if matched else "Unknown",
            "confidence": round(float(best_similarity * 100.0), 1) if matched else 0.0,
            "matched": matched,
            "bbox": best_bbox,
            "embedding": best_embedding,
            "status": "recognized" if matched else "unknown",
            "similarity": round(float(best_similarity), 3),
        }

    def process_frame(self, frame: Optional[np.ndarray], camera_name: str = "Camera") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Annotate a frame with face recognition results and return both the frame and the result."""
        if frame is None or frame.size == 0:
            return frame, self._unknown_result()

        annotated = frame.copy()
        result = self.recognize_frame(frame)
        bbox = result.get("bbox")

        if bbox is not None and result.get("status") != "no_face":
            x, y, w, h = bbox
            color = (0, 220, 0) if result.get("matched") else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            if result.get("matched"):
                label = f"{result['employee_id']} | {result['employee_name']}"
                confidence_text = f"{result['confidence']:.1f}%"
                cv2.putText(
                    annotated,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    confidence_text,
                    (x, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                logger.info(
                    "[%s] Recognized: %s | %s | Confidence: %.1f%%",
                    camera_name,
                    result["employee_id"],
                    result["employee_name"],
                    result["confidence"],
                )
            else:
                cv2.putText(
                    annotated,
                    "Unknown",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                logger.info(
                    "[%s] Unknown face detected. Best match similarity: %.4f for %s (threshold: %.2f)",
                    camera_name,
                    result.get("similarity", 0.0),
                    result.get("best_employee_id", "None"),
                    self._threshold,
                )

        return annotated, result

    def _build_employee_embedding(self, image_paths: list[str]) -> Optional[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for image_path in image_paths:
            image = self._read_image(image_path)
            if image is None:
                continue
            detections = self._detect_faces(image)
            if not detections:
                continue
            embedding = detections[0].get("embedding")
            if embedding is None:
                embedding = self._build_embedding(detections[0].get("face"))
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            return None
        return np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)

    def _read_image(self, image_path: str) -> Optional[np.ndarray]:
        path = Path(image_path)
        if not path.exists():
            return None
        image = cv2.imread(str(path))
        return image if image is not None else None

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return []

        if self._backend_name == "insightface" and self._insightface_app is not None:
            try:
                detections: List[Dict[str, Any]] = []
                faces = self._insightface_app.get(frame)
                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    face_region = frame[y1:y2, x1:x2]
                    if face_region.size == 0:
                        continue
                    embedding = getattr(face, "embedding", None)
                    if embedding is not None:
                        embedding = np.asarray(embedding, dtype=np.float32)
                    detections.append({
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "face": face_region,
                        "embedding": embedding,
                    })
                if detections:
                    return detections
            except Exception as exc:
                logger.warning("InsightFace face detection failed: %s", exc)

        if self._detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                detections = []
                for x, y, w, h in faces:
                    padding = 12
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    face_region = frame[y1:y2, x1:x2]
                    if face_region.size == 0:
                        continue
                    detections.append({
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "face": face_region,
                        "embedding": None,
                    })
                if detections:
                    return detections

        if frame.mean() < 25.0:
            return []

        return [{"bbox": (0, 0, frame.shape[1], frame.shape[0]), "face": frame, "embedding": None}]

    def _build_embedding(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        resized = cv2.GaussianBlur(resized, (5, 5), 0)
        embedding = resized.astype(np.float32).reshape(-1) / 255.0
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            return None
        return embedding / norm

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def _load_cache(self) -> Dict[str, Any]:
        if not self._cache_path.exists():
            return {}
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, OSError):
            logger.warning("Unable to read face embedding cache; rebuilding embeddings.")
        return {}

    def _save_cache(self) -> None:
        payload: Dict[str, Any] = {}
        for emp_id, record in self._employee_embeddings.items():
            payload[emp_id] = {
                "name": record.get("name", emp_id),
                "embedding": record.get("embedding", []).tolist(),
                "image_count": record.get("image_count", 0),
                "images_metadata": record.get("images_metadata", []),
            }
        try:
            self._cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to save face embedding cache: %s", exc)

    def _unknown_result(self) -> Dict[str, Any]:
        return {
            "employee_id": None,
            "employee_name": "Unknown",
            "confidence": 0.0,
            "matched": False,
            "bbox": None,
            "embedding": None,
            "status": "unknown",
        }

    def _debug_print(self, message: str) -> None:
        if not self._debug:
            return
        print("----------------------------------")
        print(message)
        print("----------------------------------")
