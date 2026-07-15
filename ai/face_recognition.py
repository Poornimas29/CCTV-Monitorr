# ai/face_recognition.py
"""InsightFace-based face recognition engine for registered employees.

This module uses exclusively the InsightFace library to run face detection,
alignment, and 512-dimensional embedding generation. It enforces a strict
512-dimension constraint and avoids any legacy fallback mock embeddings.
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
    """Loads employee reference images and recognizes faces using InsightFace."""

    def __init__(
        self,
        project_root: Optional[str] = None,
        cache_path: Optional[str] = None,
        threshold: float = 0.40,
        debug: bool = False,
    ) -> None:
        self._project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
        self._cache_path = Path(cache_path or self._project_root / ".cache" / "face_embeddings.json").resolve()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._employee_embeddings: Dict[str, Dict[str, Any]] = {}
        self._employee_lookup: Dict[str, Dict[str, Any]] = {}
        self._insightface_app = None

        # Load threshold from environment or parameter
        env_threshold = os.getenv("FACE_RECOGNITION_THRESHOLD")
        if env_threshold is not None:
            try:
                self._threshold = float(env_threshold)
            except ValueError:
                self._threshold = threshold
        else:
            self._threshold = threshold

        self._debug = debug

        # Enforce InsightFace backend setup
        self._initialize_backend()

        # Delete stale/legacy cache at startup to force rebuild using InsightFace
        if self._cache_path.exists():
            try:
                self._cache_path.unlink()
                logger.info("Deleted stale face embeddings cache file: %s", self._cache_path)
            except Exception as exc:
                logger.warning("Could not delete cache file: %s", exc)

    def _initialize_backend(self) -> None:
        """Initialize the InsightFace application. Fail if unavailable."""
        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            providers = ort.get_available_providers()
            logger.info("Available ONNX Runtime providers: %s", providers)
            selected_providers = []
            if "CUDAExecutionProvider" in providers:
                selected_providers.append("CUDAExecutionProvider")
            if "DmlExecutionProvider" in providers:
                selected_providers.append("DmlExecutionProvider")
            selected_providers.append("CPUExecutionProvider")

            logger.info("InsightFace initializing with providers: %s", selected_providers)
            app = FaceAnalysis(name="buffalo_l", providers=selected_providers)
            # Use GPU ctx_id=0 if GPU provider (CUDA/Dml) is enabled, else -1 for CPU fallback
            has_gpu = "CUDAExecutionProvider" in selected_providers or "DmlExecutionProvider" in selected_providers
            ctx_id = 0 if has_gpu else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._insightface_app = app
            logger.info("InsightFace backend loaded successfully using device context %d", ctx_id)
        except Exception as exc:
            raise RuntimeError(f"InsightFace backend is required but failed to load: {exc}")

    def initialize(self, employee_manager: Any) -> Dict[str, Any]:
        """Load employee reference metadata, generate 512-d embeddings, and cache them."""
        employees = employee_manager.get_all_employees()
        self._employee_lookup = {emp["employee_id"]: emp for emp in employees}

        cache_payload = self._load_cache()
        self._employee_embeddings = {}

        for employee in employees:
            emp_id = employee["employee_id"]
            image_paths = employee_manager.get_employee_images(emp_id)
            cache_entry = cache_payload.get(emp_id)

            # Build metadata for verification
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

            is_cache_valid = False
            if cache_entry and cache_entry.get("images_metadata") and len(cache_entry["images_metadata"]) == len(current_metadata):
                matches = True
                for c_meta, r_meta in zip(current_metadata, cache_entry["images_metadata"]):
                    if (c_meta["path"] != r_meta.get("path") or 
                        abs(c_meta["mtime"] - r_meta.get("mtime", 0.0)) > 0.1 or 
                        c_meta["size"] != r_meta.get("size")):
                        matches = False
                        break
                if matches:
                    embedding = np.asarray(cache_entry.get("embedding", []), dtype=np.float32)
                    if embedding.size == 512:
                        is_cache_valid = True
                        self._employee_embeddings[emp_id] = {
                            "employee_id": emp_id,
                            "name": employee.get("name", emp_id),
                            "embedding": embedding,
                            "image_count": len(image_paths),
                            "images_metadata": current_metadata,
                        }
                    else:
                        logger.warning(
                            "Cached embedding for '%s' had invalid dimension (%d), rebuilding.",
                            emp_id, embedding.size
                        )

            if is_cache_valid:
                continue

            embedding = self._build_employee_embedding(image_paths)
            if embedding is None:
                continue

            if embedding.size != 512:
                raise ValueError(
                    f"Generated embedding for employee '{emp_id}' is of size {embedding.size}, expected exactly 512 dimensions."
                )

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
        """Identify the best-matching employee from a camera frame."""
        if frame is None or frame.size == 0:
            return {**self._unknown_result(), "status": "no_face"}

        detections = self._detect_faces(frame)
        if not detections:
            return {**self._unknown_result(), "bbox": None, "status": "no_face"}

        if self._debug:
            self._run_diagnostics(frame, None, detections)

        best_employee_id: Optional[str] = None
        best_name: str = "Unknown"
        best_similarity = 0.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_embedding: Optional[np.ndarray] = None

        for detection in detections:
            embedding = detection.get("embedding")
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

        if self._debug:
            self._run_diagnostics(frame, crop, detections)

        best_employee_id: Optional[str] = None
        best_name: str = "Unknown"
        best_similarity = 0.0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_embedding: Optional[np.ndarray] = None

        for detection in detections:
            embedding = detection.get("embedding")
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

    def _run_diagnostics(
        self,
        frame: Optional[np.ndarray],
        crop: Optional[np.ndarray],
        detections: List[Dict[str, Any]],
    ) -> None:
        """Detailed prints. Only runs when debug=True."""
        if not self._debug:
            return

        if not detections:
            print("\n----------------------")
            print("Detected Face: No")
            print("[DIAGNOSTIC EXPLANATION]")
            print("- Reason: No face detected in the image/crop.")
            print("----------------------\n")
            return

        for idx, det in enumerate(detections):
            print("\n----------------------")
            print("Detected Face: Yes")

            embedding = det.get("embedding")
            emb_dim = len(embedding) if embedding is not None else 0
            print(f"Embedding Size: {emb_dim}")

            face_crop = det.get("face")

            best_employee_id = None
            best_similarity = -1.0

            if embedding is not None and self._employee_embeddings:
                for emp_id, record in self._employee_embeddings.items():
                    sim = self._cosine_similarity(embedding, record["embedding"])
                    print(f"{emp_id} : {sim:.4f}")
                    if sim > best_similarity:
                        best_similarity = sim
                        best_employee_id = emp_id

            matched = best_employee_id is not None and best_similarity >= self._threshold
            decision = "Recognized" if matched else "Unknown"

            print(f"Highest Match : {best_employee_id or 'None'}")
            print(f"Threshold : {self._threshold:.2f}")
            print(f"Decision : {decision}")

            if not matched:
                print("\n[DIAGNOSTIC EXPLANATION]")
                if embedding is None:
                    print("- Reason: Embedding generation failed completely.")
                elif emb_dim != 512:
                    print(f"- Reason: Embedding dimension mismatch ({emb_dim} vs expected 512).")
                elif face_crop is not None and (face_crop.shape[1] < 45 or face_crop.shape[0] < 45):
                    print(f"- Reason: Detected face is too small ({face_crop.shape[1]}x{face_crop.shape[0]} px).")
                elif best_employee_id is None:
                    print("- Reason: No registered employee matched.")
                elif best_similarity < self._threshold:
                    print(f"- Reason: Closest match is {best_employee_id} but similarity score ({best_similarity:.4f}) is below threshold ({self._threshold:.2f}).")
            print("----------------------\n")

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

    def _build_employee_embedding(self, image_paths: list[str]) -> Optional[np.ndarray]:
        """Load and extract a 512-dimension face embedding from multiple paths."""
        embeddings: list[np.ndarray] = []
        for image_path in image_paths:
            image = self._read_image(image_path)
            if image is None:
                continue
            detections = self._detect_faces(image)
            if not detections:
                continue
            embedding = detections[0].get("embedding")
            if embedding is not None:
                if embedding.size != 512:
                    raise ValueError(f"Extracted embedding dimension is {embedding.size}, expected exactly 512.")
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
        """Run face analysis using InsightFace and yield 512-d embeddings."""
        if frame is None or frame.size == 0:
            return []

        if self._insightface_app is None:
            raise RuntimeError("InsightFace application is not loaded.")

        try:
            detections: List[Dict[str, Any]] = []
            faces = self._insightface_app.get(frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue
                embedding = getattr(face, "embedding", None)
                if embedding is not None:
                    embedding = np.asarray(embedding, dtype=np.float32)
                    if embedding.size != 512:
                        raise ValueError(f"InsightFace embedding dimension is {embedding.size}, expected exactly 512.")
                detections.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "face": face_region,
                    "embedding": embedding,
                    "aimg": getattr(face, "aimg", None),
                })
            return detections
        except Exception as exc:
            logger.error("InsightFace face analysis failed: %s", exc)
            raise

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
