# detection/yolo26_detector.py
"""YOLO detector using the official Ultralytics API.

This module provides a `YOLO26Detector` class that loads a YOLO model using the
official Ultralytics API. It runs inference for person and cell phone classes.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List

import config.settings as settings

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
        
        # Detect best available device for YOLO (defaulting to CPU to prevent startup latency blocking RTSP camera threads)
        if torch.cuda.is_available():
            self.device = "cuda"
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
            "Detection Classes: [0, 67, 80, 81]\n"
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
        # Use low conf to allow thresholding in settings
        results = self.model(frame, conf=0.15, verbose=False, imgsz=320, device=self.device)[0]
        detections: List[Detection] = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Filter detections based on configurable thresholds
                if cls_idx == 0:
                    if conf < settings.CONF_PERSON:
                        continue
                    label = "person"
                elif cls_idx == 67:
                    if conf < settings.CONF_PHONE:
                        continue
                    label = "cell_phone"
                else:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=conf,
                        class_id=cls_idx,
                        label=label,
                        track_candidate=(cls_idx == 0)
                    )
                )

        # ── Color Heuristic Segmenter for Uniform and Safety Cap ──────────
        # For each person detected, analyze the crop for Uniform / Safety Cap
        frame_h, frame_w = frame.shape[:2]
        persons = [d for d in detections if d.class_id == 0]
        
        for person in persons:
            px1, py1, px2, py2 = person.bbox
            pw = px2 - px1
            ph = py2 - py1
            if pw <= 0 or ph <= 0:
                continue

            crop = frame[max(0, py1):min(frame_h, py2), max(0, px1):min(frame_w, px2)]
            if crop.size == 0:
                continue

            # Convert cropped person region to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            crop_h, crop_w = crop.shape[:2]

            # 1. Safety Cap Check (Top 20% of person's height)
            head_h = int(crop_h * 0.2)
            if head_h > 0:
                head_crop = hsv[0:head_h, :]
                total_head_pixels = head_crop.shape[0] * head_crop.shape[1]

                # HSV ranges for Safety Caps: Yellow/Orange, Blue, White
                yellow_mask = cv2.inRange(head_crop, np.array([10, 80, 80]), np.array([35, 255, 255]))
                blue_mask = cv2.inRange(head_crop, np.array([90, 50, 50]), np.array([130, 255, 255]))
                white_mask = cv2.inRange(head_crop, np.array([0, 0, 180]), np.array([180, 40, 255]))
                combined_cap_mask = yellow_mask | blue_mask | white_mask
                
                matching_pixels = cv2.countNonZero(combined_cap_mask)
                match_ratio = matching_pixels / total_head_pixels if total_head_pixels > 0 else 0.0

                if match_ratio > 0.08:  # If more than 8% matches
                    cap_conf = min(1.0, 0.5 + match_ratio * 3.0)
                    if cap_conf >= settings.CONF_SAFETY_CAP:
                        cx1 = px1
                        cy1 = py1
                        cx2 = px2
                        cy2 = py1 + head_h
                        detections.append(
                            Detection(
                                bbox=[cx1, cy1, cx2, cy2],
                                confidence=cap_conf,
                                class_id=81,
                                label="safety_cap",
                                track_candidate=False
                            )
                        )

            # 2. Uniform Check (Torso: 20% to 70% of person's height)
            torso_start_y = int(crop_h * 0.2)
            torso_end_y = int(crop_h * 0.7)
            if torso_end_y > torso_start_y:
                torso_crop = hsv[torso_start_y:torso_end_y, :]
                total_torso_pixels = torso_crop.shape[0] * torso_crop.shape[1]

                # HSV ranges for Uniforms: Blue, White, Orange/Red
                blue_mask = cv2.inRange(torso_crop, np.array([90, 50, 50]), np.array([130, 255, 255]))
                white_mask = cv2.inRange(torso_crop, np.array([0, 0, 150]), np.array([180, 40, 255]))
                orange_mask = cv2.inRange(torso_crop, np.array([0, 70, 70]), np.array([15, 255, 255]))
                combined_uniform_mask = blue_mask | white_mask | orange_mask

                matching_pixels = cv2.countNonZero(combined_uniform_mask)
                match_ratio = matching_pixels / total_torso_pixels if total_torso_pixels > 0 else 0.0

                if match_ratio > 0.15:  # If more than 15% matches
                    uniform_conf = min(1.0, 0.5 + match_ratio * 2.0)
                    if uniform_conf >= settings.CONF_UNIFORM:
                        ux1 = px1
                        uy1 = py1 + torso_start_y
                        ux2 = px2
                        uy2 = py1 + torso_end_y
                        detections.append(
                            Detection(
                                bbox=[ux1, uy1, ux2, uy2],
                                confidence=uniform_conf,
                                class_id=80,
                                label="uniform",
                                track_candidate=False
                            )
                        )

        return detections

