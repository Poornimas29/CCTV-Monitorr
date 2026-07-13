# detection/yolo26_detector.py
"""YOLO‑26 person detector (CPU‑only singleton).

This module provides a `YOLO26Detector` class that loads a YOLO‑26 model
once and re‑uses it for every frame.  The implementation uses the
Ultralytics YOLOv5 loader as a placeholder – replace the `torch.hub`
call with the actual YOLO‑26 loading code when the model file is
available.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    """Simple detection container returned by the detector.

    Attributes
    ----------
    bbox: List[int]
        Bounding box as [x1, y1, x2, y2] in pixel coordinates.
    confidence: float
        Confidence score from the model.
    track_candidate: bool
        Flag used by the tracker – always ``True`` for person detections.
    """
    bbox: List[int]
    confidence: float
    track_candidate: bool = True

class YOLO26Detector:
    """CPU‑only singleton detector.

    Parameters
    ----------
    model_path: str
        Path to the ``.pt`` or ``.onnx`` file containing the YOLO‑26 weights.
    """
    _instance = None

    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str):
        # Initialise only once even if called multiple times.
        if getattr(self, "_initialized", False):
            return
        self.model_path = Path(model_path)
        self.device = "cpu"
        # ----- Placeholder loader -----
        # The Ultralytics YOLOv5 repo can load a custom model file that
        # follows the YOLO format.  Replace this with an actual YOLO‑26 load
        # routine when the model is available.
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(self.model_path),
            force_reload=False,
        ).to(self.device)
        # Optional: adjust confidence threshold.
        self.model.conf = 0.4
        self._initialized = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame and return only person detections.

        Parameters
        ----------
        frame: np.ndarray
            Image in BGR format as returned by ``cv2.VideoCapture``.

        Returns
        -------
        List[Detection]
            Detections where ``class_id == 0`` (person) according to COCO
            ordering used by YOLO‑26.
        """
        results = self.model(frame)
        detections: List[Detection] = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            # COCO class 0 corresponds to "person".
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append(
                Detection(bbox=[x1, y1, x2, y2], confidence=float(conf))
            )
        return detections
