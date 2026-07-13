# detection/yolo26_detector.py
"""YOLO-26 person and phone detector (CPU-only singleton).

This module provides a `YOLO26Detector` class that loads a YOLO-26 model
once and re-uses it for every frame. If torch or the model file is not available,
it falls back to a simulated detection mode to keep the system runnable.
"""

import logging
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Simple detection container returned by the detector.

    Attributes
    ----------
    bbox: List[int]
        Bounding box as [x1, y1, x2, y2] in pixel coordinates.
    confidence: float
        Confidence score from the model.
    class_id: int
        COCO class ID (0 for person, 67 for cell phone).
    label: str
        Label name ("person" or "cell_phone").
    track_candidate: bool
        Flag used by the tracker - always True for person detections.
    """
    bbox: List[int]
    confidence: float
    class_id: int = 0
    label: str = "person"
    track_candidate: bool = True


class YOLO26Detector:
    """CPU-only singleton detector."""
    _instance = None

    def __new__(cls, model_path: str = ""):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls, model_path: str = "") -> 'YOLO26Detector':
        """Access the singleton instance of the detector."""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path: str = ""):
        # Initialise only once even if called multiple times.
        if getattr(self, "_initialized", False):
            return
        
        self.model_path = Path(model_path) if model_path else None
        self.device = "cpu"
        self.simulation = True
        self.model = None

        try:
            import torch
            if self.model_path and self.model_path.exists():
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(self.model_path),
                    force_reload=False,
                ).to(self.device)
                self.model.conf = 0.4
                self.simulation = False
                logger.info("YOLO26Detector initialized with PyTorch model at: %s", self.model_path)
            else:
                logger.warning("YOLO model path not specified or does not exist. Falling back to simulation mode.")
        except Exception as exc:
            logger.warning("Could not load PyTorch YOLO detector: %s. Running in simulation mode.", exc)

        self._initialized = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame and return person and phone detections."""
        if self.simulation or self.model is None:
            return self._simulate_detections(frame)

        try:
            results = self.model(frame)
            detections: List[Detection] = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                cls_idx = int(cls)
                if cls_idx == 0:
                    label = "person"
                elif cls_idx == 67:
                    label = "cell_phone"
                else:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=float(conf),
                        class_id=cls_idx,
                        label=label,
                        track_candidate=(cls_idx == 0)
                    )
                )
            return detections
        except Exception as exc:
            logger.error("YOLO inference failed: %s. Falling back to simulation.", exc)
            return self._simulate_detections(frame)

    def _simulate_detections(self, frame: np.ndarray) -> List[Detection]:
        """Generate consistent simulated person and cell phone detections."""
        if frame is None or frame.size == 0:
            return []
        
        h, w = frame.shape[:2]
        t = time.time()
        detections: List[Detection] = []

        # Simulate Person 1: Walking across the left/center of the screen
        x1 = int((w * 0.2) + (w * 0.1) * np.sin(t * 0.4))
        y1 = int((h * 0.3) + 20 * np.cos(t * 0.4))
        x2 = x1 + int(w * 0.2)
        y2 = y1 + int(h * 0.5)

        # Clip bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        detections.append(
            Detection(bbox=[x1, y1, x2, y2], confidence=0.88, class_id=0, label="person")
        )

        # Simulate Person 2: Static on the right side of the screen
        x3 = int(w * 0.65)
        y3 = int(h * 0.25)
        x4 = x3 + int(w * 0.18)
        y4 = y3 + int(h * 0.6)

        x3, y3 = max(0, x3), max(0, y3)
        x4, y4 = min(w, x4), min(h, y4)

        detections.append(
            Detection(bbox=[x3, y3, x4, y4], confidence=0.82, class_id=0, label="person")
        )

        # Simulate a Phone Usage for Person 1 (active for 4 seconds, inactive for 4 seconds)
        if int(t) % 8 < 4:
            px_center = (x1 + x2) // 2
            py_center = (y1 + y2) // 2
            
            # Draw phone near hand/chest area
            ph_x1 = px_center - 15
            ph_y1 = py_center - 10
            ph_x2 = px_center + 15
            ph_y2 = py_center + 10
            
            detections.append(
                Detection(
                    bbox=[ph_x1, ph_y1, ph_x2, ph_y2],
                    confidence=0.76,
                    class_id=67,
                    label="cell_phone",
                    track_candidate=False
                )
            )

        return detections
