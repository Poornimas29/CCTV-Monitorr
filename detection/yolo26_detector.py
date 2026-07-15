# detection/yolo26_detector.py
"""YOLO detector using the official Ultralytics API.

This module provides a `YOLO26Detector` class that loads a YOLO model using the
official Ultralytics API. It runs inference for person and cell phone classes.
"""

import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Simple detection container returned by the detector."""
    bbox: List[int]
    confidence: float
    class_id: int = 0
    label: str = "person"
    track_candidate: bool = True


class YOLO26Detector:
    """Singleton detector using the official Ultralytics API."""
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

    def __init__(self, model_path: str = "") -> None:
        if getattr(self, "_initialized", False):
            return

        import ultralytics
        from ultralytics import YOLO
        import torch
        import openvino as ov

        self.model_path = Path(model_path) if model_path else None
        
        # Detect best available device for YOLO
        core = ov.Core()
        if torch.cuda.is_available():
            self.device = "cuda"
        elif "GPU" in core.available_devices:
            self.device = "intel:gpu"
        else:
            self.device = "cpu"

        # Load custom model if provided, otherwise default to yolo26n.pt
        model_name = str(self.model_path) if self.model_path else "yolo26n.pt"

        # If targeting OpenVINO GPU, ensure model is exported to OpenVINO format
        if self.device == "intel:gpu":
            base_model_name = model_name
            openvino_dir = Path(model_name).stem + "_openvino_model"
            if not Path(openvino_dir).exists():
                logger.info("Exporting YOLO model to OpenVINO format for GPU acceleration...")
                temp_model = YOLO(base_model_name)
                temp_model.export(format="openvino", imgsz=320)
            model_name = openvino_dir

        try:
            self.model = YOLO(model_name, task="detect")
            if self.device in ["cpu", "cuda"]:
                self.model.to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}")

        # Print YOLO Detector Startup Report to Console exactly as requested
        report = (
            "\n--------------------------------\n"
            "Detection Engine\n\n"
            f"Model File: {model_name}\n"
            "Framework: OpenVINO (Ultralytics API)" if self.device == "intel:gpu" else "Framework: PyTorch (Ultralytics API)",
            f"Ultralytics Version: {ultralytics.__version__}\n"
            f"Device: {self.device}\n"
            "Detection Classes: [0, 67]\n"
            "Tracking: Class 0 (Person) Only passed to ByteTrack\n"
            "--------------------------------"
        )
        # Format list to string if report is a tuple
        report_str = "\n".join(report) if isinstance(report, tuple) else report
        print(report_str, flush=True)
        logger.info(report_str)

        self._initialized = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame and return person and phone detections."""
        if frame is None or frame.size == 0:
            return []

        # Run inference using ultralytics YOLO model on selected device (CPU/GPU/CUDA) with imgsz=320
        results = self.model(frame, conf=0.4, verbose=False, imgsz=320, device=self.device)[0]
        detections: List[Detection] = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_idx = int(box.cls[0].item())
                if cls_idx == 0:
                    label = "person"
                elif cls_idx == 67:
                    label = "cell_phone"
                else:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=conf,
                        class_id=cls_idx,
                        label=label,
                        track_candidate=(cls_idx == 0)
                    )
                )
        return detections
