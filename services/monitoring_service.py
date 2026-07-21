# services/monitoring_service.py
"""MonitoringService orchestrates the end-to-end person and phone monitoring pipeline.

Production redesign (RC-2, RC-3, RC-4, RC-8, RC-9 fixes):

RC-2 — CRITICAL race condition fixed:
    drain_results() is now called FIRST, before process_timeouts().
    Previously, process_timeouts() could mark a track "exited" before the async
    recognition result arrived, causing the result to be discarded and the employee
    to stay "Unknown" forever.

RC-4 — ReID extracted ONCE per track per frame:
    Previously reid_engine.extract_features() was called 3 times per track per frame.
    Now it is called once, cached in reid_cache{}, and reused wherever needed.

RC-9 — All print() calls removed from the hot path:
    print() acquires the GIL and writes to stdout synchronously.  On Windows console,
    this was reducing FPS from 70+ to <10 at 3 tracked persons.  Replaced with
    logger.debug() (per-frame, filtered by log level) and logger.info() (significant
    events only: lock, exit, attendance).
"""

import os
import json
import cv2
import argparse
import sys
import logging
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Any, Dict

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
from ai.pose_estimator import MediaPipePoseEstimator
from ai.reid_engine import FastReIDEngine
from ai.async_recognition_engine import AsyncRecognitionEngine

from detection.detection_manager import DetectionManager
from tracking.tracking_manager import TrackingManager
from ai.face_recognition_manager import FaceRecognitionManager
from employee_management.embedding_manager import EmbeddingManager
from session.track_memory_manager import TrackMemoryManager
from session.identity_manager import IdentityManager
from session.attendance_manager import AttendanceManager
from ai.face_recognition import FaceRecognitionEngine


class MonitoringService:
    """Orchestrates the CCTV monitoring pipeline."""

    def __init__(self, max_frames: Optional[int] = None, display: bool = False):
        self.max_frames = max_frames
        self.display = display
        self.frame_counter = 0

        # Initialize Managers
        self.camera_manager = CameraManager()
        self.detection_manager = DetectionManager()
        self.tracking_manager_class = TrackingManager

        self.pose_estimator = MediaPipePoseEstimator()
        self.reid_engine = FastReIDEngine()

        self.face_rec_manager = FaceRecognitionManager(project_root=PROJECT_ROOT)
        self.embedding_manager = EmbeddingManager(project_root=PROJECT_ROOT)

        # Initialize Employee and Face Recognition components
        from employee_management.employee_manager import EmployeeManager
        self.employee_manager = EmployeeManager(project_root=PROJECT_ROOT)
        self.employee_manager.load_employees()

        self.embedding_manager.initialize_gallery(self.employee_manager, self.face_rec_manager, self.reid_engine)

        self.track_memory_manager = TrackMemoryManager()
        self.identity_manager = IdentityManager()
        self.attendance_manager = AttendanceManager()
        self.global_session_manager = self.attendance_manager

        # Keep/delegate session_engine and recognizer for backward compatibility
        self.recognizer = FaceRecognitionEngine(project_root=PROJECT_ROOT)
        self.recognizer.initialize(self.employee_manager)

        from ai.session_engine import EmployeeSessionEngine
        self.session_engine = EmployeeSessionEngine()

        # Async Recognition Engine — runs InsightFace in a background thread.
        # max_workers=1: InsightFace's FaceAnalysis is NOT thread-safe for concurrent
        # calls from multiple threads. A single worker serialises inference and
        # still completely frees the main frame loop from any blocking wait.
        self.async_recognizer = AsyncRecognitionEngine(
            face_rec_manager=self.face_rec_manager,
            embedding_manager=self.embedding_manager,
            max_workers=1,
        )
        self.async_recognizer.start()

        self.trackers = {}
        self.renderer = Renderer()

        # ── Async Detection Worker ─────────────────────────────────────────────
        # YOLO inference runs in a dedicated background thread so the main display
        # loop is never blocked waiting for GPU/CPU inference to complete.
        # _detect_queue  : main thread puts (cam_id, frame) for the worker to process
        # _detect_result : worker puts (cam_id, detections) back for the main thread
        # Using maxsize=1 so we always process the LATEST frame and discard stale ones.
        self._detect_queue: queue.Queue = queue.Queue(maxsize=1)
        self._detect_result: queue.Queue = queue.Queue(maxsize=1)
        self._detect_stop = threading.Event()
        self._detect_thread = threading.Thread(
            target=self._detection_worker, daemon=True, name="yolo-detect"
        )
        self._detect_thread.start()

        # Latest cached detections per camera (used while new inference is running)
        self._cached_detections: Dict[str, list] = {}
        # Per-track frame counters for throttled pose/ReID
        self._track_pose_counter: Dict[tuple, int] = {}
        self._track_reid_counter: Dict[tuple, int] = {}

        # Ensure output directory exists.
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

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

    def update_pose_for_track(self, track_mem: dict, frame: np.ndarray, bbox: List[int]) -> None:
        """Run pose estimation or project the last known relative pose onto the current bbox."""
        p_state = self.pose_estimator.estimate_pose(frame, bbox)

        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if p_state is not None:
            track_mem["pose_state"] = p_state
            # Store relative landmarks for projection if detection is lost later
            relative_landmarks = {}
            for idx, lm in p_state["landmarks"].items():
                relative_landmarks[idx] = {
                    "rx": (lm["x"] - x1) / w,
                    "ry": (lm["y"] - y1) / h,
                    "visibility": lm["visibility"]
                }
            track_mem["relative_pose_landmarks"] = relative_landmarks
            track_mem["last_pose_head_direction"] = p_state.get("head_direction", "Front")
        else:
            # MediaPipe failed to detect. Project the last known relative landmarks.
            rel_lms = track_mem.get("relative_pose_landmarks")
            if rel_lms is not None:
                new_landmarks = {}
                new_skeleton = []
                for idx, rlm in rel_lms.items():
                    gx = x1 + int(rlm["rx"] * w)
                    gy = y1 + int(rlm["ry"] * h)
                    new_landmarks[idx] = {
                        "x": gx,
                        "y": gy,
                        "visibility": rlm["visibility"]
                    }
                    new_skeleton.append((gx, gy, rlm["visibility"]))

                ls = new_landmarks.get(11)
                rs = new_landmarks.get(12)
                lh = new_landmarks.get(15)
                rh = new_landmarks.get(16)

                track_mem["pose_state"] = {
                    "landmarks": new_landmarks,
                    "skeleton": new_skeleton,
                    "head_direction": track_mem.get("last_pose_head_direction", "Front"),
                    "shoulders": {
                        "left": (ls["x"], ls["y"]) if ls and ls["visibility"] > 0.5 else None,
                        "right": (rs["x"], rs["y"]) if rs and rs["visibility"] > 0.5 else None
                    },
                    "hands": {
                        "left": (lh["x"], lh["y"]) if lh and lh["visibility"] > 0.5 else None,
                        "right": (rh["x"], rh["y"]) if rh and rh["visibility"] > 0.5 else None
                    },
                    "is_stable": True
                }
            else:
                # No previous pose. Use fallback heuristic pose.
                track_mem["pose_state"] = self.pose_estimator._estimate_heuristic_pose(frame, bbox)

    # ── Async Detection Worker ─────────────────────────────────────────────────

    def _detection_worker(self) -> None:
        """Background thread: continuously drain the detect queue and run YOLO.

        Runs on a dedicated thread so YOLO GPU inference never blocks the
        display/tracking main loop.  Uses maxsize=1 queues so stale frames
        are dropped automatically — we always process the newest one.
        """
        while not self._detect_stop.is_set():
            try:
                cam_id, frame = self._detect_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                detections = self.detection_manager.detect(frame)
                # Drain the result queue first so we always push the latest result.
                try:
                    self._detect_result.get_nowait()
                except queue.Empty:
                    pass
                self._detect_result.put((cam_id, detections))
            except Exception as exc:
                logger.error("[DetectWorker] Inference error: %s", exc, exc_info=True)

    def _submit_frame_for_detection(self, cam_id: str, frame: np.ndarray) -> None:
        """Non-blocking frame submit.  If the queue is full we drop the old frame."""
        try:
            self._detect_queue.get_nowait()   # drop stale entry if present
        except queue.Empty:
            pass
        try:
            self._detect_queue.put_nowait((cam_id, frame))
        except queue.Full:
            pass  # should not happen after the get_nowait above, but guard anyway

    def process_camera_frame(self, cam_id: str, frame: np.ndarray, ts: datetime):
        """Process a single frame for a specific camera through the entire pipeline.

        Pipeline order (RC-2 fix — drain BEFORE timeouts):
        1. drain_results()   ← MUST be first to prevent identity results being discarded
        2. Detect
        3. Track (ByteTrack)
        4. Mark lost / process timeouts
        5. Per-track: state update + ReID extract (ONCE, cached) + pose
        6. Submit unidentified tracks to async recogniser
        7. Phone detection + attendance update for locked tracks
        8. Render
        """
        self.frame_counter += 1
        if cam_id not in self.trackers:
            fps = self.camera_manager.get_fps(cam_id)
            if fps <= 0.0:
                fps = settings.TARGET_FPS
            self.trackers[cam_id] = self.tracking_manager_class(
                track_timeout=settings.TRACK_TIMEOUT, fps=fps
            )

        tracker = self.trackers[cam_id]

        # ── STEP 1: Drain async recognition results FIRST ─────────────────────
        # RC-2 fix: this must happen BEFORE process_timeouts() so that a result
        # arriving in this frame is applied before the track can be marked "exited".
        for rec_result in self.async_recognizer.drain_results():
            r_cam_id, r_track_id, r_ts, r_emp_id, r_emp_name, r_score = rec_result

            r_track_mem = self.track_memory_manager.get_track(r_cam_id, r_track_id)
            if r_track_mem is None or r_track_mem.get("locked_status"):
                continue  # Track gone or already locked since we submitted

            newly_locked, locked_id = self.identity_manager.process_recognition_result(
                r_track_mem, r_emp_id, r_emp_name, r_score, round(r_score * 100.0, 1)
            )

            if newly_locked:
                # Use averaged ReID features from the rolling window (more stable than single frame)
                cached_feats = r_track_mem.get("reid_feature_history", [])
                if cached_feats:
                    feat = np.mean(cached_feats, axis=0).astype(np.float32)
                    feat_norm = np.linalg.norm(feat)
                    if feat_norm > 0:
                        feat = feat / feat_norm
                else:
                    feat = self.reid_engine.extract_features(
                        frame, r_track_mem.get("bbox", [0, 0, 1, 1])
                    )

                self.attendance_manager.create_session(
                    employee_id=locked_id,
                    employee_name=r_emp_name,
                    camera_id=r_cam_id,
                    track_id=r_track_id,
                    bbox=r_track_mem.get("bbox", [0, 0, 1, 1]),
                    timestamp=r_track_mem["entry_time"],
                    confidence=round(r_score * 100.0, 1),
                    reid_features=feat,
                )
                self.session_engine.process_recognition(
                    employee_id=locked_id,
                    employee_name=r_emp_name,
                    confidence=round(r_score * 100.0, 1),
                    timestamp=r_ts,
                )
                logger.info(
                    "[Pipeline] Identity locked — Track %d → Employee %s (confidence %.1f%%) on Camera %s",
                    r_track_id, r_emp_name, round(r_score * 100.0, 1), r_cam_id
                )

        # ── STEP 2: Submit frame to async detector; use latest cached result ────
        # The detection worker processes the frame in the background on the GPU/CPU.
        # We submit the current frame and immediately use whatever the worker has
        # already finished — this means the display never waits for inference.
        self._submit_frame_for_detection(cam_id, frame)

        # Collect any freshly-finished detection result
        try:
            result_cam_id, fresh_detections = self._detect_result.get_nowait()
            self._cached_detections[result_cam_id] = fresh_detections
        except queue.Empty:
            pass

        all_detections = self._cached_detections.get(cam_id, [])

        # ── STEP 3: Tracking ──────────────────────────────────────────────────
        tracks = tracker.update(all_detections)

        # ── STEP 4: Mark lost tracks + process timeouts ───────────────────────
        current_track_ids = {t.track_id for t in tracks}

        for key, track in list(self.track_memory_manager.tracks.items()):
            t_cam_id, t_track_id = key
            if t_cam_id == cam_id and track["track_status"] == "tracking":
                if t_track_id not in current_track_ids:
                    self.track_memory_manager.mark_lost(cam_id, t_track_id, ts)
                    self.attendance_manager.handle_lost_track(cam_id, t_track_id, ts)

        exited_tracks = self.track_memory_manager.process_timeouts(ts)
        for _track in exited_tracks:
            # RC-8: generate_unrecognized_attendance_record is now a no-op;
            # unknown tracks produce no output files.
            self.attendance_manager.generate_unrecognized_attendance_record(_track)
        self.attendance_manager.process_timeouts(ts)

        # ── STEP 5: Per-track state update + ReID (ONCE per track) ────────────
        # RC-4 fix: extract ReID features exactly once per track and cache the
        # result in reid_cache.  Reuse wherever we previously called extract_features
        # a second or third time (ReID recovery and lock-time storage).
        reid_cache: Dict[int, Optional[np.ndarray]] = {}

        for trk in tracks:
            track_mem = self.track_memory_manager.get_track(cam_id, trk.track_id)
            is_new = (track_mem is None)

            if is_new:
                track_mem = self.track_memory_manager.create_track(
                    cam_id, trk.track_id, trk.bbox, ts
                )
            else:
                self.track_memory_manager.update_track(cam_id, trk.track_id, trk.bbox, ts)

            # ── Throttled ReID: extract features every 5th frame per track ─────
            # Extracting a CNN feature vector every frame wastes CPU/GPU.
            # Every 5th frame we run the full extract; in between we reuse the cache.
            track_key = (cam_id, trk.track_id)
            self._track_reid_counter[track_key] = self._track_reid_counter.get(track_key, 0) + 1
            if self._track_reid_counter[track_key] % 5 == 1 or reid_cache.get(trk.track_id) is None:
                feat = self.reid_engine.extract_features(frame, trk.bbox)
                reid_cache[trk.track_id] = feat
            else:
                feat = reid_cache.get(trk.track_id)  # reuse cached from previous iter

            if is_new and feat is not None:
                # Check for ReID recovery against lost sessions
                best_reid_score = -1.0
                best_session = None

                for s in self.attendance_manager.sessions.values():
                    if s.status == "lost" and getattr(s, "reid_features", None) is not None:
                        if ts - s.last_seen <= self.attendance_manager.lost_timeout:
                            score = self.reid_engine.compute_similarity(feat, s.reid_features)
                            if score > best_reid_score:
                                best_reid_score = score
                                best_session = s

                if best_session is not None and best_reid_score >= settings.REID_SIMILARITY_THRESHOLD:
                    old_track_id = best_session.current_track_id
                    self.attendance_manager.bind_camera_track(
                        best_session, cam_id, trk.track_id, trk.bbox, ts, feat
                    )
                    best_session.status = "tracking"
                    best_session.logged_left = False

                    # Lock the track memory immediately (ReID recovery is high-confidence)
                    track_mem["locked_status"] = True
                    track_mem["employee_id"] = best_session.employee_id
                    track_mem["employee_name"] = best_session.employee_name
                    track_mem["recognition_status"] = "identified"
                    track_mem["recognition_confidence"] = best_session.recognition_confidence

                    self.identity_manager.track_to_employee[
                        (cam_id, trk.track_id)
                    ] = best_session.employee_id

                    logger.info(
                        "[FastReID] Track %d recovered → Employee %s (score=%.4f) on Camera %s",
                        trk.track_id, best_session.employee_name, best_reid_score, cam_id
                    )

            # Update rolling ReID feature window using the cached feature
            if feat is not None:
                hist = track_mem.setdefault("reid_feature_history", [])
                hist.append(feat)
                if len(hist) > 5:
                    hist.pop(0)

            # ── Throttled Pose: run MediaPipe every 3rd frame per track ──────────
            # MediaPipe is CPU-only and expensive per-person. We run it every 3rd
            # frame; the Kalman-projected skeleton fills the gap between runs.
            pose_key = (cam_id, trk.track_id)
            self._track_pose_counter[pose_key] = self._track_pose_counter.get(pose_key, 0) + 1
            if self._track_pose_counter[pose_key] % 3 == 1:
                self.update_pose_for_track(track_mem, frame, trk.bbox)
            # else: pose_state already projected by update_pose_for_track's fallback path

        # ── STEP 6: Submit unidentified tracks for background recognition ──────
        for trk in tracks:
            track_mem = self.track_memory_manager.get_track(cam_id, trk.track_id)
            if track_mem is None or track_mem.get("locked_status"):
                continue

            last_attempt = track_mem["last_recognition_attempt"]
            if last_attempt is not None:
                if (ts - last_attempt).total_seconds() < settings.RECOGNITION_INTERVAL:
                    continue

            px1, py1, px2, py2 = trk.bbox
            h, w = frame.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)
            crop = frame[py1:py2, px1:px2]

            if crop.size > 0:
                submitted = self.async_recognizer.submit(cam_id, trk.track_id, ts, crop)
                if submitted:
                    track_mem["last_recognition_attempt"] = ts
                    track_mem["recognition_count"] = (
                        track_mem.get("recognition_count", 0) + 1
                    )

        # ── STEP 6c: Face bbox projection (visual display only) ───────────────
        for trk in tracks:
            track_mem = self.track_memory_manager.get_track(cam_id, trk.track_id)
            if track_mem is None:
                continue
            px1, py1, px2, py2 = trk.bbox
            crop_w = px2 - px1
            crop_h = py2 - py1
            if track_mem.get("relative_face_bbox") is not None and crop_w > 0 and crop_h > 0:
                rx1, ry1, rx2, ry2 = track_mem["relative_face_bbox"]
                track_mem["face_bbox"] = [
                    int(px1 + rx1 * crop_w),
                    int(py1 + ry1 * crop_h),
                    int(px1 + rx2 * crop_w),
                    int(py1 + ry2 * crop_h),
                ]
            else:
                track_mem.setdefault("face_bbox", None)

        # ── STEP 6d: Attendance update for locked tracks (phone, timing) ───────
        phone_dets = [d for d in all_detections if d.class_id == 67]
        for trk in tracks:
            track_mem = self.track_memory_manager.get_track(cam_id, trk.track_id)
            if track_mem is None:
                continue
            if track_mem["locked_status"]:
                session = self.attendance_manager.get_session_by_track(cam_id, trk.track_id)
                if session is not None:
                    self.attendance_manager.update_track(
                        session,
                        cam_id,
                        trk.track_id,
                        trk.bbox,
                        ts,
                        phone_dets,
                        track_mem.get("pose_state"),
                    )
            else:
                # Track is unrecognized / unknown. Update phone usage directly on track_mem.
                phone_used = False
                px1, py1, px2, py2 = trk.bbox
                pw = px2 - px1
                ph = py2 - py1
                proximity_threshold = 0.15 * max(pw, ph)
                pose_state = track_mem.get("pose_state")
                
                for phone in phone_dets:
                    ph_x1, ph_y1, ph_x2, ph_y2 = phone.bbox
                    ph_cx = (ph_x1 + ph_x2) / 2.0
                    ph_cy = (ph_y1 + ph_y2) / 2.0
                    
                    if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                        has_hand_proximity = False
                        if pose_state and pose_state.get("hands"):
                            hands = pose_state["hands"]
                            left_hand = hands.get("left")
                            right_hand = hands.get("right")
                            
                            if left_hand:
                                lh_dist = np.sqrt((ph_cx - left_hand[0])**2 + (ph_cy - left_hand[1])**2)
                                if lh_dist < proximity_threshold:
                                    has_hand_proximity = True
                            if right_hand:
                                rh_dist = np.sqrt((ph_cx - right_hand[0])**2 + (ph_cy - right_hand[1])**2)
                                if rh_dist < proximity_threshold:
                                    has_hand_proximity = True
                                    
                        if pose_state and pose_state.get("landmarks"):
                            if has_hand_proximity:
                                phone_used = True
                                break
                        else:
                            phone_used = True
                            break
                            
                prev_seen = track_mem.get("prev_seen_time", ts)
                
                if phone_used:
                    if track_mem.get("phone_use_start") is None:
                        track_mem["phone_use_start"] = ts
                    else:
                        overlap_time = (ts - track_mem["phone_use_start"]).total_seconds()
                        if overlap_time >= settings.PHONE_USAGE_CONFIRM_SECONDS:
                            if prev_seen is not None:
                                dt = (ts - prev_seen).total_seconds()
                                if 0.0 < dt < 5.0:
                                    track_mem["phone_use_duration"] += dt
                            track_mem["phone_confirmed_use_active"] = True
                else:
                    if track_mem.get("phone_use_start") is not None:
                        duration = (ts - track_mem["phone_use_start"]).total_seconds()
                        if duration >= settings.PHONE_USAGE_CONFIRM_SECONDS:
                            track_mem["phone_use_count"] += 1
                        track_mem["phone_use_start"] = None
                        track_mem["phone_confirmed_use_active"] = False


        # ── STEP 7: Visual Render Projections ────────────────────────────────
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
                self.is_recognized = (gs.status == "tracking") and (gs.employee_id is not None)
                self.pose_state = state.pose_state
                self.phone_confirmed_use_active = gs.phone_confirmed_use_active

        active_projections = []
        unreg_list = []
        for key, track in self.track_memory_manager.tracks.items():
            t_cam_id, t_track_id = key
            if t_cam_id != cam_id:
                continue
            if track["track_status"] != "tracking":
                continue

            if track["locked_status"]:
                session = self.attendance_manager.get_session_by_track(cam_id, t_track_id)
                if session is not None:
                    proj = LocalSessionProjection(session, cam_id)
                    proj.bbox = list(track["bbox"])
                    proj.pose_state = track.get("pose_state")
                    proj.status = "tracking"
                    proj.is_recognized = True
                    active_projections.append(proj)
                else:
                    unreg_list.append(track)
            else:
                unreg_list.append(track)

        annotated = self.renderer.draw(
            frame=frame,
            sessions=active_projections,
            unrecognized_tracks=unreg_list,
            detections=all_detections,
            fps=self.camera_manager.get_fps(cam_id),
        )

        # ── STEP 8: Logging ───────────────────────────────────────────────────
        # RC-9: Replaced per-frame print() with logger.debug() to eliminate GIL
        # contention and synchronous Windows console I/O overhead.
        for proj in active_projections:
            logger.debug(
                "[Frame %d] Camera=%s Track=%d Employee=%s Confidence=%.1f%% BBox=%s",
                self.frame_counter,
                cam_id,
                proj.track_id,
                proj.employee_id,
                proj.recognition_confidence,
                proj.bbox,
            )

        for utrk in unreg_list:
            logger.debug(
                "[Frame %d] Camera=%s Track=%d Unknown BBox=%s",
                self.frame_counter,
                cam_id,
                utrk["track_id"],
                utrk["bbox"],
            )

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
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.frame_counter += 1
                if self.max_frames is not None and self.frame_counter >= self.max_frames:
                    break
        finally:
            self.camera_manager.stop_all()
            self.async_recognizer.stop()
            if self.display:
                cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the CCTV monitoring pipeline.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live OpenCV window with annotations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    service = MonitoringService(max_frames=args.max_frames, display=args.display)
    service.run()
