import os
import sys
import cv2
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
os.environ["RTSP_URL"] = ""

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from stream.camera_manager import CameraManager
from employee_management.employee_manager import EmployeeManager
from ai.face_recognition import FaceRecognitionEngine

def recheck():
    print("=" * 70)
    print("               RECHECKING CHANNEL 6 (CAM002)                 ")
    print("=" * 70)
    
    # 1. Initialize
    employee_manager = EmployeeManager(project_root=PROJECT_ROOT)
    employee_manager.load_employees()
    
    recognizer = FaceRecognitionEngine(project_root=PROJECT_ROOT, threshold=0.40, debug=True)
    recognizer.initialize(employee_manager)
    
    # Start cameras
    manager = CameraManager()
    manager.start_all()
    print("Starting streams... Waiting 5 seconds...")
    time.sleep(5)
    
    # Try to read multiple frames from CAM002
    print("Reading frames from CAM002...")
    frame = None
    for i in range(15):
        time.sleep(0.2)
        frame = manager.get_latest_frame("CAM002")
        if frame is not None:
            break
            
    if frame is None:
        print("[FAIL] Unable to read any frames from CAM002")
        manager.stop_all()
        return
        
    print(f"Captured frame shape: {frame.shape}")
    
    # Save original frame
    cv2.imwrite("debug/original_cam2.jpg", frame)
    
    # Run face detection & recognition
    detections = recognizer._detect_faces(frame)
    if not detections:
        print("[FAIL] Face detection failed on CAM002 frame. No face detected.")
        manager.stop_all()
        return
        
    print(f"[SUCCESS] Detected {len(detections)} face(s) in CAM002 frame!")
    
    for idx, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        face_crop = det["face"]
        cv2.imwrite(f"debug/detected_face_cam2.jpg", face_crop)
        
        embedding = det.get("embedding")
        if embedding is None:
            embedding = recognizer._build_embedding(face_crop)
            
        if embedding is None:
            print(f"Failed to build embedding for face {idx}")
            continue
            
        print(f"\nFace #{idx} BBox: {det['bbox']}")
        print("-" * 50)
        
        best_sim = -1.0
        best_emp = None
        for emp_id, record in recognizer._employee_embeddings.items():
            sim = recognizer._cosine_similarity(embedding, record["embedding"])
            print(f"  {emp_id} ({record['name']}) Similarity: {sim:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_emp = emp_id
                
        threshold = recognizer._threshold
        print(f"  Best Match: {best_emp} (Similarity: {best_sim:.4f})")
        print(f"  Threshold: {threshold:.2f}")
        decision = "MATCHED" if best_sim >= threshold else "UNKNOWN"
        print(f"  Decision: {decision}")
        print("-" * 50)
        
        # Annotate frame
        annotated_frame = frame.copy()
        color = (40, 220, 40) if decision == "MATCHED" else (0, 165, 255)
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        label = f"{best_emp if decision == 'MATCHED' else 'Unknown'} ({best_sim:.2f})"
        cv2.putText(annotated_frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite("debug/annotated_cam2.jpg", annotated_frame)
        print("Saved annotated frame to debug/annotated_cam2.jpg")
        
    manager.stop_all()
    print("=" * 70)

if __name__ == "__main__":
    recheck()
