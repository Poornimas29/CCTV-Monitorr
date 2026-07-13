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
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

logger = logging.getLogger(__name__)

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

        # Initialize per-camera state managers
        self.trackers = {}
        self.person_managers = {}
        
        # Initialise shared pipeline modules
        self.camera_manager = CameraManager()
        self.detector = YOLO26Detector.instance()
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

    def _compute_appearance_histogram(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
            
        ch, cw = person_crop.shape[:2]
        # Focus on the torso (middle 60% vertically and 80% horizontally) to represent clothing
        ty1, ty2 = int(ch * 0.2), int(ch * 0.8)
        tx1, tx2 = int(cw * 0.1), int(cw * 0.9)
        torso_crop = person_crop[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return None
            
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

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

    def process_camera_frame(self, cam_id: str, frame: np.ndarray, ts: datetime):
        """Process a single frame for a specific camera through the entire pipeline."""
        if cam_id not in self.trackers:
            self.trackers[cam_id] = Tracker()
            self.person_managers[cam_id] = PersonManager()
            
        tracker = self.trackers[cam_id]
        person_manager = self.person_managers[cam_id]

        # 1. Detection - get both persons and phone detections
        all_detections = self.detector.detect(frame)
        
        # Separate detections
        person_dets = [d for d in all_detections if d.class_id == 0]
        phone_dets = [d for d in all_detections if d.class_id == 67]

        # 2. Tracking - track persons persistently
        tracks = tracker.update(person_dets)

        # 3. Update track lifecycle and perform phone overlap checks
        person_states = person_manager.process_tracks_with_phones(
            camera_id=cam_id,
            timestamp=ts,
            tracks=tracks,
            phone_detections=phone_dets
        )

        # 4. Face ReID and Face Recognition on Cropped regions
        for p in person_states:
            if p.status == "tracking":
                # Periodically update appearance histogram
                if getattr(p, "reid_hist", None) is None or p.frame_count % 10 == 0:
                    p.reid_hist = self._compute_appearance_histogram(frame, p.bbox)

                # Try ReID on first frame of a new track
                if p.frame_count == 1 and p.reid_hist is not None:
                    best_reid_score = -1.0
                    best_candidate = None
                    best_candidate_dist = 0.0

                    p_cx = (p.bbox[0] + p.bbox[2]) / 2.0
                    p_cy = (p.bbox[1] + p.bbox[3]) / 2.0

                    for other in person_manager._persons.values():
                        if other.track_id != p.track_id and other.status in ("lost", "exited") and getattr(other, "reid_hist", None) is not None:
                            if ts - other.last_seen <= timedelta(seconds=15):
                                other_cx = (other.bbox[0] + other.bbox[2]) / 2.0
                                other_cy = (other.bbox[1] + other.bbox[3]) / 2.0
                                dist = np.sqrt((p_cx - other_cx)**2 + (p_cy - other_cy)**2)

                                if dist < 400.0:
                                    score = cv2.compareHist(p.reid_hist, other.reid_hist, cv2.HISTCMP_CORREL)
                                    if score > best_reid_score:
                                        best_reid_score = score
                                        best_candidate = other
                                        best_candidate_dist = dist

                    if best_candidate is not None and best_reid_score >= 0.85:
                        p.recognition_status = best_candidate.recognition_status
                        p.employee_id = best_candidate.employee_id
                        p.employee_name = best_candidate.employee_name
                        p.recognition_confidence = getattr(best_candidate, "recognition_confidence", 0.0)
                        p.session_start_time = getattr(best_candidate, "session_start_time", None)
                        p.phone_use_duration = best_candidate.phone_use_duration
                        p.productivity_score = best_candidate.productivity_score
                        p.last_recognition_attempt = getattr(best_candidate, "last_recognition_attempt", None)
                        
                        # Mark the merged candidate as exited so it's not matched again
                        best_candidate.status = "exited"
                        
                        logger.info(
                            "[ReID] Track ID %d merged with recently lost Track ID %d (%s) - Score: %.4f, Dist: %.1f px",
                            p.track_id, best_candidate.track_id, p.employee_name, best_reid_score, best_candidate_dist
                        )

                # Crop region only if they are not yet identified
                if p.recognition_status != "identified":
                    # Check if retry interval has elapsed (5 seconds)
                    should_attempt = False
                    last_attempt = getattr(p, "last_recognition_attempt", None)
                    if last_attempt is None:
                        should_attempt = True
                    elif (ts - last_attempt).total_seconds() >= 5.0:
                        should_attempt = True

                    if should_attempt:
                        p.last_recognition_attempt = ts
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
                                p.recognition_confidence = face_res["confidence"]
                                p.session_start_time = ts
                                
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
        return annotated, person_states

    def run(self):
        """Start the CCTV monitoring loop."""
        self.camera_manager.start_all()
        try:
            for cam_id, frame, ts in self.camera_manager.read_frames():
                if frame is None or frame.size == 0:
                    continue

                annotated, person_states = self.process_camera_frame(cam_id, frame, ts)

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
