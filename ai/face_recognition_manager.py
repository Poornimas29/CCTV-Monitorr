# ai/face_recognition_manager.py
"""FaceRecognitionManager manages InsightFace inference and face quality validation."""

import os
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import config.settings as settings

logger = logging.getLogger(__name__)

class FaceRecognitionManager:
    """Handles face detection, pose validation, blur checking, and embedding generation using InsightFace."""
    
    def __init__(self, project_root: Optional[str] = None, debug: bool = False) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.insightface_app = None
        self.debug = debug
        self._initialize_backend()

    def _initialize_backend(self) -> None:
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
            has_gpu = "CUDAExecutionProvider" in selected_providers or "DmlExecutionProvider" in selected_providers
            ctx_id = 0 if has_gpu else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.insightface_app = app
            logger.info("InsightFace backend loaded successfully using device context %d", ctx_id)
        except Exception as exc:
            raise RuntimeError(f"InsightFace backend failed to load: {exc}")

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run face analysis using InsightFace and yield 512-d embeddings."""
        if frame is None or frame.size == 0:
            return []

        try:
            detections = []
            faces = self.insightface_app.get(frame)
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
                
                detections.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "face": face_region,
                    "embedding": embedding,
                    "pose": getattr(face, "pose", None),
                    "kps": getattr(face, "kps", None),
                    "face_obj": face
                })
            return detections
        except Exception as exc:
            logger.error("InsightFace face analysis failed: %s", exc)
            return []

    def validate_face(self, face_det: Dict[str, Any], min_size: int = None, min_quality: float = None) -> Tuple[bool, str]:
        """Validate if the detected face is suitable for recognition based on size, quality, blur, and angle."""
        # 1. Face size check
        fx, fy, fw, fh = face_det["bbox"]
        target_min_size = min_size if min_size is not None else settings.MIN_FACE_SIZE
        if fw < target_min_size or fh < target_min_size:
            return False, f"Face size too small ({fw}x{fh} < {target_min_size})"

        # 2. Quality/blur check (Laplacian variance)
        face_crop = face_det["face"]
        if face_crop.size == 0:
            return False, "Empty face crop"
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        target_min_quality = min_quality if min_quality is not None else settings.MIN_FACE_QUALITY
        if variance < target_min_quality:
            return False, f"Face too blurred (variance {variance:.1f} < {target_min_quality})"

        # 3. Angle check (yaw, pitch, roll)
        pose = face_det.get("pose")
        if pose is not None:
            pitch, yaw, roll = pose
            # CCTV cameras are ceiling-mounted and look DOWN at employees.
            # This naturally produces high pitch angles (top-of-head views).
            # Pitch limit is raised to 55° to accept these valid frames.
            # Yaw stays at 45°: true profile faces don't embed well regardless of camera height.
            if abs(yaw) > 45.0 or abs(pitch) > 55.0 or abs(roll) > 30.0:
                return False, f"Face angle out of bounds (yaw:{yaw:.1f}, pitch:{pitch:.1f}, roll:{roll:.1f})"
        else:
            # Fallback to keypoints symmetry check
            kps = face_det.get("kps")
            if kps is not None and len(kps) == 5:
                lex, ley = kps[0]
                rex, rey = kps[1]
                nx, ny = kps[2]
                eye_dist = max(1.0, abs(rex - lex))
                left_dist = abs(nx - lex)
                right_dist = abs(nx - rex)
                ratio = abs(left_dist - right_dist) / eye_dist
                if ratio > 0.55:
                    return False, f"Face turned too far (symmetry ratio {ratio:.2f} > 0.55)"

        return True, "Face is acceptable"
