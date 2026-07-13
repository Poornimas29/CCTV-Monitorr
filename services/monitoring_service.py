# services/monitoring_service.py
"""MonitoringService orchestrates the end-to-end person and phone monitoring pipeline.

It ties together:
* CameraManager - reads frames from RTSP streams.
* YOLO26Detector - detects persons and mobile phones.
* Tracker - tracks persons persistently.
* PersonManager - manages track lifecycles, checks phone overlaps, and calculates productivity.
* FaceRecognitionEngine - recognizes faces on cropped person regions.
* Renderer - annotates frames.
"""

import os
import json
import cv2
import argparse
import sys
from datetime import datetime
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import config.settings as settings
from stream.camera_manager import CameraManager
from detection.yolo26_detector import YOLO26Detector
from tracking.tracker import Tracker
from person_management.person_manager import PersonManager
from visualisation.renderer import Renderer


class MonitoringService:
    """Orchestrates the CCTV monitoring pipeline."""
    def __init__(self, max_frames: Optional[int] = None, display: bool = False):
        self.max_frames = max_frames
        self.display = display
        self.frame_counter = 0

        # Initialise core pipeline modules
        self.camera_manager = CameraManager()
        self.detector = YOLO26Detector.instance()
        self.tracker = Tracker()
        self.person_manager = PersonManager()
        self.renderer = Renderer()

        # Initialize Employee and Face Recognition components
        from employee_management.employee_manager import EmployeeManager
        self.employee_manager = EmployeeManager(project_root=PROJECT_ROOT)
        self.employee_manager.load_employees()

        from ai.face_recognition import FaceRecognitionEngine
        self.recognizer = FaceRecognitionEngine(project_root=PROJECT_ROOT)
        self.recognizer.initialize(self.employee_manager)

        from ai.session_engine import EmployeeSessionEngine
        self.session_engine = EmployeeSessionEngine()

        # Ensure output directory exists.
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    def _write_json(self, cam_id: str, ts: datetime, persons: list):
        person_list = []
        for p in persons:
            person_list.append({
                "track_id": p.track_id,
                "bbox": p.bbox,
                "status": p.status,
                "recognition_status": p.recognition_status,
                "employee_id": p.employee_id,
                "employee_name": p.employee_name,
                "phone_use_detected": p.phone_use_detected,
                "phone_use_duration": round(p.phone_use_duration, 2),
                "total_tracked_duration": round(p.total_tracked_duration, 2),
                "productivity_score": round(p.productivity_score, 1)
            })

        payload = {
            "camera_id": cam_id,
            "timestamp": ts.isoformat(),
            "persons": person_list,
        }
        filename = f"{cam_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.json"
        out_path = os.path.join(settings.OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self):
        """Start the CCTV monitoring loop."""
        self.camera_manager.start_all()
        try:
            for cam_id, frame, ts in self.camera_manager.read_frames():
                if frame is None or frame.size == 0:
                    continue

                # 1. Detection - get both persons and phone detections
                all_detections = self.detector.detect(frame)
                
                # Separate detections
                person_dets = [d for d in all_detections if d.class_id == 0]
                phone_dets = [d for d in all_detections if d.class_id == 67]

                # 2. Tracking - track persons persistently
                tracks = self.tracker.update(person_dets)

                # 3. Update track lifecycle and perform phone overlap checks
                person_states = self.person_manager.process_tracks_with_phones(
                    camera_id=cam_id,
                    timestamp=ts,
                    tracks=tracks,
                    phone_detections=phone_dets
                )

                # 4. Face Recognition on Cropped regions for unidentified tracks
                for p in person_states:
                    if p.status == "tracking":
                        # Crop region only if they are not yet identified
                        if p.recognition_status != "identified":
                            px1, py1, px2, py2 = p.bbox
                            h, w = frame.shape[:2]
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(w, px2), min(h, py2)

                            crop = frame[py1:py2, px1:px2]
                            if crop.size > 0:
                                face_res = self.recognizer.recognize_crop(crop)
                                if face_res.get("matched"):
                                    p.recognition_status = "identified"
                                    p.employee_id = face_res["employee_id"]
                                    p.employee_name = face_res["employee_name"]
                                    
                                    # Trigger attendance logging
                                    self.session_engine.process_recognition(
                                        employee_id=p.employee_id,
                                        employee_name=p.employee_name,
                                        confidence=face_res["confidence"],
                                        timestamp=ts
                                    )
                                else:
                                    p.recognition_status = "unknown"
                                    p.employee_name = "Unknown"
                                    p.employee_id = None
                        
                        # Re-verify matching state mapping
                        if p.recognition_status != "identified":
                            p.employee_name = "Unknown"
                            p.employee_id = None

                # 5. Visual Render
                fps = self.camera_manager.get_fps(cam_id)
                annotated = self.renderer.draw(
                    frame=frame,
                    persons=person_states,
                    fps=fps,
                )

                # 6. JSON output logging
                self._write_json(cam_id, ts, person_states)

                if self.display:
                    cv2.imshow(f"{cam_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.frame_counter += 1
                if self.max_frames is not None and self.frame_counter >= self.max_frames:
                    break
        finally:
            self.camera_manager.stop_all()
            if self.display:
                cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the CCTV monitoring pipeline.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process before exiting.")
    parser.add_argument("--display", action="store_true",
                        help="Show a live OpenCV window with annotations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    service = MonitoringService(max_frames=args.max_frames, display=args.display)
    service.run()
