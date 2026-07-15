# scratch/test_onnx.py
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

print("Available providers:", ort.get_available_providers())

print("Loading ONNX model...")
onnx_model = YOLO("yolov8n.onnx", task="detect")

print("ONNX model device:", onnx_model.device)

print("Running inference...")
dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
results = onnx_model(dummy_frame)
print("ONNX Inference Success! Output boxes count:", len(results[0].boxes) if results[0].boxes is not None else 0)
