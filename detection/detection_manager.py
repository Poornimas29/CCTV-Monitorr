# detection/detection_manager.py
"""DetectionManager class wraps YOLO26Detector for standard detection flow."""

import numpy as np
from typing import List
from detection.yolo26_detector import YOLO26Detector, Detection

class DetectionManager:
    """Manages the object detection phase of the pipeline."""
    def __init__(self, model_path: str = "") -> None:
        self.detector = YOLO26Detector.instance(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs yolo detection and heuristics on the input frame."""
        return self.detector.detect(frame)
