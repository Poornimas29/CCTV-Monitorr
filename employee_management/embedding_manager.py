# employee_management/embedding_manager.py
"""EmbeddingManager manages employee embedding galleries and matches embeddings."""

import os
import json
import logging
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import config.settings as settings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages database of employee face embeddings and handles gallery matching."""
    
    def __init__(self, project_root: str, cache_path: Optional[str] = None) -> None:
        self.project_root = Path(project_root)
        self.cache_path = Path(cache_path or self.project_root / ".cache" / "face_embeddings.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.employee_embeddings: Dict[str, Dict[str, Any]] = {}
        # Pre-computed gallery matrices for vectorised matching (RC-7)
        self._face_gallery: Optional[np.ndarray] = None
        self._face_gallery_ids: List[str] = []
        self._face_gallery_names: List[str] = []
        self._reid_gallery: Optional[np.ndarray] = None
        self._reid_gallery_ids: List[str] = []
        self._reid_gallery_names: List[str] = []

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def load_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.warning("Unable to read face embedding cache; rebuilding. Error: %s", exc)
        return {}

    def save_cache(self) -> None:
        payload: Dict[str, Any] = {}
        for emp_id, record in self.employee_embeddings.items():
            payload[emp_id] = {
                "name": record["name"],
                "embeddings": [emb.tolist() for emb in record["embeddings"]],
                "reid_embeddings": [emb.tolist() for emb in record.get("reid_embeddings", [])],
                "image_count": record["image_count"],
                "images_metadata": record["images_metadata"]
            }
        try:
            self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Unable to save face embedding cache: %s", exc)

    def initialize_gallery(self, employee_manager: Any, face_recognition_manager: Any, reid_engine: Optional[Any] = None) -> Dict[str, Any]:
        """Loads employee reference images, extracts multiple face & ReID embeddings, and caches them."""
        employees = employee_manager.get_all_employees()
        cache_payload = self.load_cache()
        self.employee_embeddings = {}

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
            # Validate cache entry has "embeddings" (plural) and matches metadata
            if cache_entry and cache_entry.get("images_metadata") and len(cache_entry["images_metadata"]) == len(current_metadata):
                matches = True
                for c_meta, r_meta in zip(current_metadata, cache_entry["images_metadata"]):
                    if (c_meta["path"] != r_meta.get("path") or 
                        abs(c_meta["mtime"] - r_meta.get("mtime", 0.0)) > 0.1 or 
                        c_meta["size"] != r_meta.get("size")):
                        matches = False
                        break
                
                cached_embeddings_list = cache_entry.get("embeddings", [])
                cached_reid_list = cache_entry.get("reid_embeddings", [])
                # Backward compatibility check for single "embedding" field
                if not cached_embeddings_list and "embedding" in cache_entry:
                    cached_embeddings_list = [cache_entry["embedding"]]
                
                if matches and cached_embeddings_list:
                    embeddings = [np.asarray(emb, dtype=np.float32) for emb in cached_embeddings_list]
                    reid_embeddings = [np.asarray(emb, dtype=np.float32) for emb in cached_reid_list]
                    if all(emb.size == 512 for emb in embeddings):
                        is_cache_valid = True
                        self.employee_embeddings[emp_id] = {
                            "employee_id": emp_id,
                            "name": employee.get("name", emp_id),
                            "embeddings": embeddings,
                            "reid_embeddings": reid_embeddings,
                            "image_count": len(image_paths),
                            "images_metadata": current_metadata,
                        }
                    else:
                        logger.warning("Cached embeddings for '%s' had invalid dimension, rebuilding.", emp_id)

            if is_cache_valid:
                continue

            # Rebuild embeddings list
            embeddings = []
            reid_embeddings = []
            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                # Extract face detections
                faces = face_recognition_manager.detect_faces(image)
                if faces:
                    embedding = faces[0].get("embedding")
                    if embedding is not None and embedding.size == 512:
                        embeddings.append(embedding)
                        logger.info("Embedding Generated for %s from image %s", emp_id, Path(image_path).name)

                # Extract Re-ID body features
                if reid_engine is not None:
                    h_img, w_img = image.shape[:2]
                    reid_feat = reid_engine.extract_features(image, [0, 0, w_img, h_img])
                    if reid_feat is not None:
                        reid_embeddings.append(reid_feat)

            if not embeddings and not reid_embeddings:
                continue

            self.employee_embeddings[emp_id] = {
                "employee_id": emp_id,
                "name": employee.get("name", emp_id),
                "embeddings": embeddings,
                "reid_embeddings": reid_embeddings,
                "image_count": len(image_paths),
                "images_metadata": current_metadata,
            }

        self.save_cache()
        # Build vectorised gallery matrices for fast matching (RC-7)
        self._build_gallery_matrix()
        return {
            "registered_employees": len(self.employee_embeddings),
            "total_face_images": sum(len(employee_manager.get_employee_images(emp_id)) for emp_id in self.employee_embeddings),
            "embeddings_loaded": bool(self.employee_embeddings),
        }

    def _build_gallery_matrix(self) -> None:
        """Pre-compute contiguous numpy matrices for O(1) vectorised embedding matching.

        RC-7 fix: replaces O(N) Python loops in match_embedding with a single
        matrix-vector dot product: gallery_matrix @ query_vector.
        RC-10 fix: ensures all stored embeddings are explicitly L2-normalised so
        that the dot product is equivalent to cosine similarity.
        """
        face_embs: List[np.ndarray] = []
        face_ids: List[str] = []
        face_names: List[str] = []
        reid_embs: List[np.ndarray] = []
        reid_ids: List[str] = []
        reid_names: List[str] = []

        for emp_id, record in self.employee_embeddings.items():
            name = record["name"]
            for emb in record.get("embeddings", []):
                if emb is None or emb.size != 512:
                    continue
                emb = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                face_embs.append(emb / norm if norm > 0 else emb)
                face_ids.append(emp_id)
                face_names.append(name)
            for emb in record.get("reid_embeddings", []):
                if emb is None or not isinstance(emb, np.ndarray) or emb.ndim != 1:
                    continue
                emb = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(emb)
                reid_embs.append(emb / norm if norm > 0 else emb)
                reid_ids.append(emp_id)
                reid_names.append(name)

        if face_embs:
            self._face_gallery = np.stack(face_embs, axis=0).astype(np.float32)  # (N, 512)
            self._face_gallery_ids = face_ids
            self._face_gallery_names = face_names
        else:
            self._face_gallery = None
            self._face_gallery_ids = []
            self._face_gallery_names = []

        if reid_embs:
            self._reid_gallery = np.stack(reid_embs, axis=0).astype(np.float32)
            self._reid_gallery_ids = reid_ids
            self._reid_gallery_names = reid_names
        else:
            self._reid_gallery = None
            self._reid_gallery_ids = []
            self._reid_gallery_names = []

        logger.info(
            "[EmbeddingManager] Gallery matrix ready: %d face embeddings, %d ReID embeddings across %d employees.",
            len(face_embs), len(reid_embs), len(self.employee_embeddings)
        )

    def match_embedding(self, query_emb: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """Vectorised gallery match — single numpy dot product instead of a Python loop.

        RC-7 fix: gallery_matrix @ query_emb is equivalent to computing cosine
        similarity for all N gallery entries simultaneously using BLAS/MKL.
        ~100x faster than the previous Python loop for typical gallery sizes.
        """
        if self._face_gallery is None or not self._face_gallery_ids:
            return None, "Unknown", 0.0

        q = np.asarray(query_emb, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = self._face_gallery @ q          # (N,) — vectorised cosine similarity
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        return (
            self._face_gallery_ids[best_idx],
            self._face_gallery_names[best_idx],
            best_score,
        )

    def match_reid_embedding(self, query_emb: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """Vectorised ReID gallery match with shape and normalisation guards.

        RC-10 fix: adds explicit shape validation so HSV histograms (2D arrays)
        can never reach this function and cause a silent incorrect similarity score.
        """
        if self._reid_gallery is None or not self._reid_gallery_ids:
            return None, "Unknown", 0.0

        q = np.asarray(query_emb, dtype=np.float32)
        if q.ndim != 1:
            # 2D histogram from HSV fallback — cannot compare with 1D ReID vectors
            return None, "Unknown", 0.0

        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = self._reid_gallery @ q          # (M,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        return (
            self._reid_gallery_ids[best_idx],
            self._reid_gallery_names[best_idx],
            best_score,
        )
