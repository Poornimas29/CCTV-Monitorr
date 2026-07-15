# scratch/test_openvino.py
import numpy as np
from ultralytics import YOLO

print("Loading PyTorch model...")
model = YOLO("yolov8n.pt")

print("Exporting to OpenVINO...")
# Exporting format "openvino" creates a directory named yolov8n_openvino_model/
openvino_path = model.export(format="openvino", imgsz=320)
print(f"Exported to OpenVINO: {openvino_path}")

print("Loading OpenVINO model on GPU...")
ov_model = YOLO("yolov8n_openvino_model", task="detect")

print("Running inference on GPU...")
dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
# Device "GPU" directs OpenVINO to run inference on the Intel Iris Xe GPU!
results = ov_model(dummy_frame, device="GPU")
print("OpenVINO GPU Inference Success! Output boxes count:", len(results[0].boxes) if results[0].boxes is not None else 0)
