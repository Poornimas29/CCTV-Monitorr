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

        # Initialize per-camera state managers and global session manager
        from session.global_session_manager import GlobalSessionManager
        self.trackers = {}
        self.global_session_manager = GlobalSessionManager()
        self.unrecognized_tracks = {}
        self.prev_bboxes = {}
        
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

    def _write_json(self, cam_id: str, ts: datetime, sessions: list, unrecognized_tracks: list):
        person_list = []
        for s in sessions:
            person_list.append({
                "track_id": s.track_id,
                "bbox": s.bbox,
                "status": s.status,
                "recognition_status": "identified",
                "employee_id": s.employee_id,
                "employee_name": s.employee_name,
                "phone_use_detected": s.phone_use_detected,
                "phone_use_duration": round(s.phone_use_duration, 2),
                "total_tracked_duration": round(s.total_tracked_duration, 2),
                "productivity_score": round(s.productivity_score, 1)
            })
        for utrk in unrecognized_tracks:
            person_list.append({
                "track_id": utrk["track_id"],
                "bbox": utrk["bbox"],
                "status": "tracking",
                "recognition_status": "unknown",
                "employee_id": None,
                "employee_name": "Unknown",
                "phone_use_detected": False,
                "phone_use_duration": 0.0,
                "total_tracked_duration": 0.0,
                "productivity_score": 100.0
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
        self.frame_counter += 1
        if cam_id not in self.trackers:
            self.trackers[cam_id] = Tracker()
            self.unrecognized_tracks[cam_id] = {}
            
        tracker = self.trackers[cam_id]
        prev_unreg_tracks = self.unrecognized_tracks[cam_id]

        # 1. Detection - get both persons and phone detections
        all_detections = self.detector.detect(frame)
        
        # Separate detections
        person_dets = [d for d in all_detections if d.class_id == 0]
        phone_dets = [d for d in all_detections if d.class_id == 67]

        # 2. Tracking - track persons persistently
        tracks = tracker.update(person_dets)

        # 3. Update active sessions and unrecognized tracks
        current_track_ids = {t.track_id for t in tracks}
        
        # Mark visible tracks for this camera that are not in the current frame as lost
        for session in list(self.global_session_manager.sessions.values()):
            if session.status != "exited" and cam_id in session.visible_cameras:
                track_state = session.visible_cameras[cam_id]
                if track_state.track_id not in current_track_ids:
                    # Mark track as lost on this camera
                    self.global_session_manager.handle_lost_track(cam_id, track_state.track_id, ts)
                    print("----------------------")
                    print("Track Lost")
                    print(f"Track {track_state.track_id}")
                    print("Searching Lost Registry")
                    print("----------------------")
                    logger.info("Track Lost - Track ID %d, searching lost registry", track_state.track_id)

        active_unrecognized_tracks = {}

        # 4. Map track lifecycle and try ReID / Face recognition
        for trk in tracks:
            session = self.global_session_manager.get_session_by_track(cam_id, trk.track_id)
            if session is not None:
                # Update identified employee session
                self.global_session_manager.update_track(session, cam_id, trk.track_id, trk.bbox, ts, phone_dets)
            else:
                # Check if this track is in our previous unrecognized list
                prev_unreg = prev_unreg_tracks.get(trk.track_id)
                if prev_unreg is None:
                    # New Track! Try ReID recovery against lost sessions
                    recomputed_hist = self._compute_appearance_histogram(frame, trk.bbox)
                    
                    best_reid_score = -1.0
                    best_session = None

                    p_cx = (trk.bbox[0] + trk.bbox[2]) / 2.0
                    p_cy = (trk.bbox[1] + trk.bbox[3]) / 2.0

                    if recomputed_hist is not None:
                        for s in self.global_session_manager.sessions.values():
                            if s.status == "lost" and getattr(s, "reid_hist", None) is not None:
                                if ts - s.last_seen <= timedelta(seconds=30):
                                    other_cx = (s.current_bbox[0] + s.current_bbox[2]) / 2.0
                                    other_cy = (s.current_bbox[1] + s.current_bbox[3]) / 2.0
                                    dist = np.sqrt((p_cx - other_cx)**2 + (p_cy - other_cy)**2)

                                    if dist < 400.0:
                                        score = cv2.compareHist(recomputed_hist, s.reid_hist, cv2.HISTCMP_CORREL)
                                        if score > best_reid_score:
                                            best_reid_score = score
                                            best_session = s

                    if best_session is not None and best_reid_score >= 0.85:
                        old_track_id = best_session.current_track_id
                        # Reconnect track to this session globally
                        self.global_session_manager.bind_camera_track(
                            best_session, cam_id, trk.track_id, trk.bbox, ts, recomputed_hist
                        )
                        best_session.status = "tracking"
                        best_session.logged_left = False
                        
                        print("----------------------")
                        print("Track Changed")
                        print(f"Employee ID: {best_session.employee_id}")
                        print(f"Old Track ID: {old_track_id}")
                        print(f"New Track ID: {trk.track_id}")
                        print("Recovery Method: Torso ReID color histogram + spatial distance matching")
                        print("----------------------")
                        logger.info(
                            "[ReID] Reconnected Track ID %d to Employee Session %s (%s) - Score: %.4f",
                            trk.track_id, best_session.session_id, best_session.employee_name, best_reid_score
                        )
                    else:
                        # ReID match failed. Log new track creation
                        print("----------------------")
                        print("New Track Created")
                        print(f"Track {trk.track_id}")
                        print("Unknown")
                        print("----------------------")
                        logger.info("New Track Created - Track ID %d, status: Unknown", trk.track_id)
                        
                        # Add to unrecognized tracks list
                        active_unrecognized_tracks[trk.track_id] = {
                            "track_id": trk.track_id,
                            "bbox": trk.bbox,
                            "face_bbox": None,
                            "last_recognition_attempt": None
                        }
                else:
                    # Update existing unrecognized track bbox
                    prev_unreg["bbox"] = trk.bbox
                    active_unrecognized_tracks[trk.track_id] = prev_unreg

        # 5. Process face detection & recognition on unrecognized tracks
        for uid, utrk in active_unrecognized_tracks.items():
            px1, py1, px2, py2 = utrk["bbox"]
            h, w = frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            crop = frame[py1:py2, px1:px2]
            utrk["face_bbox"] = None # Reset face bbox for this frame

            if crop.size > 0:
                detections = self.recognizer._detect_faces(crop)
                if detections:
                    det = detections[0]
                    fx, fy, fw, fh = det["bbox"]
                    utrk["face_bbox"] = [px1 + fx, py1 + fy, px1 + fx + fw, py1 + fy + fh]

                    # Check if retry interval has elapsed (2 seconds)
                    should_attempt = False
                    last_attempt = utrk["last_recognition_attempt"]
                    if last_attempt is None or (ts - last_attempt).total_seconds() >= 2.0:
                        should_attempt = True

                    if should_attempt:
                        utrk["last_recognition_attempt"] = ts
                        face_res = self.recognizer.recognize_crop(crop, frame=frame)
                        if face_res.get("matched"):
                            # Face recognized!
                            emp_id = face_res["employee_id"]
                            emp_name = face_res["employee_name"]
                            confidence = face_res["confidence"]
                            rehist = self._compute_appearance_histogram(frame, utrk["bbox"])

                            session = self.global_session_manager.create_session(
                                employee_id=emp_id,
                                employee_name=emp_name,
                                camera_id=cam_id,
                                track_id=uid,
                                bbox=utrk["bbox"],
                                timestamp=ts,
                                confidence=confidence,
                                reid_hist=rehist
                            )
                            # Store the custom reid_hist on the session attribute for subsequent matches
                            session.reid_hist = rehist
                            utrk["recognized"] = True

                            # Trigger session engine logging
                            self.session_engine.process_recognition(
                                employee_id=emp_id,
                                employee_name=emp_name,
                                confidence=confidence,
                                timestamp=ts
                            )
                            print("----------------------")
                            print("Employee Session Started")
                            print(f"Employee ID: {emp_id}")
                            print(f"Employee Name: {emp_name}")
                            print(f"Track ID: {uid}")
                            print(f"Recognition Confidence: {confidence:.1f}%")
                            print(f"Recognition Time: {ts:%Y-%m-%d %H:%M:%S}")
                            print(f"Production Start Time: {ts:%Y-%m-%d %H:%M:%S}")
                            print("----------------------")
                            logger.info("Face Recognized - %s | Confidence: %.1f%% | Identity Locked", emp_name, confidence)
                        else:
                            if not utrk.get("logged_unknown", False):
                                utrk["logged_unknown"] = True
                                print("----------------------")
                                print("Unknown face detected")
                                print(f"Camera: {cam_id}")
                                print(f"Track ID: {uid}")
                                print(f"Time: {ts:%Y-%m-%d %H:%M:%S}")
                                print("----------------------")

        # Store unrecognized tracks
        self.unrecognized_tracks[cam_id] = {
            uid: utrk for uid, utrk in active_unrecognized_tracks.items()
            if not utrk.get("recognized", False)
        }

        # 6. Clean up lost sessions that exceed the lost timeout
        exited_list = self.global_session_manager.process_timeouts(ts)
        for session in exited_list:
            duration_sec = session.working_duration
            m, s = divmod(int(duration_sec), 60)
            h, m = divmod(m, 60)
            duration_str = f"{h}h {m}m {s}s"
            print("----------------------")
            print("Employee Session Ended")
            print(f"Employee ID: {session.employee_id}")
            print(f"Employee Name: {session.employee_name}")
            print(f"Production End Time: {session.last_seen:%Y-%m-%d %H:%M:%S}")
            print(f"Working Duration: {duration_str}")
            print("----------------------")
            logger.info("Employee Left - %s | Duration: %s", session.employee_name, duration_str)

        # 7. Visual Render
        fps = self.camera_manager.get_fps(cam_id)
        
        # Build Camera Session Projections
        class LocalSessionProjection:
            def __init__(self, gs, camera_id):
                state = gs.visible_cameras[camera_id]
                self.session_id = gs.session_id
                self.employee_id = gs.employee_id
                self.employee_name = gs.employee_name
                self.track_id = state.track_id
                self.bbox = state.bbox
                self.last_seen = state.last_seen
                self.first_seen = gs.first_seen
                self.status = gs.status
                self.phone_use_detected = state.phone_use_detected
                self.phone_use_duration = gs.phone_use_duration
                self.productivity_score = gs.productivity_score
                self.recognition_confidence = gs.recognition_confidence

        active_projections = []
        for s in self.global_session_manager.sessions.values():
            if s.status != "exited" and cam_id in s.visible_cameras:
                active_projections.append(LocalSessionProjection(s, cam_id))
                
        unreg_list = list(self.unrecognized_tracks[cam_id].values())
        
        annotated = self.renderer.draw(
            frame=frame,
            sessions=active_projections,
            unrecognized_tracks=unreg_list,
            fps=fps,
        )

        # Print frame-by-frame debug logs for active recognized employees on this camera
        for proj in active_projections:
            if proj.status == "tracking":
                key = (proj.session_id, cam_id)
                prev_bbox = self.prev_bboxes.get(key, proj.bbox)
                
                print("----------------------")
                print(f"Frame Number: {self.frame_counter}")
                print(f"Employee ID: {proj.employee_id}")
                print(f"Track ID: {proj.track_id}")
                print(f"Current Bounding Box: {proj.bbox}")
                print(f"Previous Bounding Box: {prev_bbox}")
                print("----------------------")
                
                self.prev_bboxes[key] = proj.bbox

        return annotated, active_projections

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
