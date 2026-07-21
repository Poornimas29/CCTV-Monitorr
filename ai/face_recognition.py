# ai/face_recognition.py
"""InsightFace-based face recognition engine facade for registered employees.

Delegates responsibilities to FaceRecognitionManager and EmbeddingManager.
Keeps full backward compatibility for existing tests and main loop.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from ai.face_recognition_manager import FaceRecognitionManager
from employee_management.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """Loads employee reference images and recognizes faces using InsightFace (Delegates to managers)."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        cache_path: Optional[str] = None,
        threshold: float = 0.40,
        debug: bool = False,
    ) -> None:
        self._project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self.embedding_manager = EmbeddingManager(self._project_root, cache_path)
        self.face_recognition_manager = FaceRecognitionManager(self._project_root, debug=debug)
        self._threshold = threshold
        self._debug = debug

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return self.embedding_manager._cosine_similarity(left, right)

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return self.face_recognition_manager.detect_faces(frame)

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Public alias to support delegation patterns."""
        return self._detect_faces(frame)

    def initialize(self, employee_manager: Any) -> Dict[str, Any]:
        """Loads employee reference metadata and builds embedding database."""
        # Delete stale/legacy cache if requested / needed
        return self.embedding_manager.initialize_gallery(employee_manager, self)

    def recognize_frame(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Identify the best-matching employee from a camera frame."""
        if frame is None or frame.size == 0:
            return {**self._unknown_result(), "status": "no_face"}

        detections = self._detect_faces(frame)
        if not detections:
            return {**self._unknown_result(), "bbox": None, "status": "no_face"}

        best_employee_id = None
        best_name = "Unknown"
        best_similarity = 0.0
        best_bbox = None
        best_embedding = None

        for detection in detections:
            embedding = detection.get("embedding")
            if embedding is None:
                continue

            emp_id, emp_name, similarity = self.embedding_manager.match_embedding(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_employee_id = emp_id
                best_name = emp_name
                best_bbox = detection.get("bbox")
                best_embedding = embedding

        matched = best_employee_id is not None and best_similarity >= self._threshold

        if self._debug:
            decision = "Recognized" if matched else "Unknown"
            debug_lines = [
                "Frame Received: Yes",
                f"Number of Faces Detected: {len(detections)}",
            ]
            if best_employee_id:
                debug_lines.append(f"{best_employee_id} : {best_similarity:.4f}")
            debug_lines.extend([
                f"Best Match: {best_employee_id or 'None'}",
                f"Recognition Threshold: {self._threshold:.2f}",
                f"Final Decision: {decision}",
            ])
            print("----------------------------------")
            print("\n".join(debug_lines))
            print("----------------------------------")

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

    def recognize_crop(self, crop: Optional[np.ndarray], frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Recognize a face within a cropped person image."""
        if crop is None or crop.size == 0:
            return self._unknown_result()

        detections = self._detect_faces(crop)
        if not detections:
            return {**self._unknown_result(), "status": "no_face"}

        best_employee_id = None
        best_name = "Unknown"
        best_similarity = 0.0
        best_bbox = None
        best_embedding = None

        for detection in detections:
            embedding = detection.get("embedding")
            if embedding is None:
                continue

            emp_id, emp_name, similarity = self.embedding_manager.match_embedding(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_employee_id = emp_id
                best_name = emp_name
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
        """Annotate a frame with face recognition details."""
        if frame is None or frame.size == 0:
            return frame, self._unknown_result()

        annotated = frame.copy()
        result = self.recognize_frame(frame)
        bbox = result.get("bbox")

        if bbox is not None and result.get("status") != "no_face":
            x, y, w, h = bbox
            color = (0, 220, 0) if result.get("matched") else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        return annotated, result

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
