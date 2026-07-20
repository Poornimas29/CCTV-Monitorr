# session/global_session_manager.py
"""GlobalSessionManager manages multi-camera employee tracking sessions,
decoupled from raw camera-specific track IDs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import config.settings as settings

@dataclass
class CameraTrackState:
    """Represents the tracking state of a session on a specific camera."""
    track_id: int
    bbox: List[int]
    last_seen: datetime
    phone_use_detected: bool = False
    pose_state: Optional[Dict[str, Any]] = None


@dataclass
class GlobalSession:
    """Represents a global tracking session of an identified employee across all cameras."""
    session_id: str
    employee_id: str
    employee_name: str
    status: str
    first_seen: datetime
    last_seen: datetime
    reid_features: Optional[np.ndarray] = None
    reid_hist: Optional[np.ndarray] = None  # Backward compatibility fallback
    visible_cameras: Dict[str, CameraTrackState] = field(default_factory=dict)
    current_track_id: int = -1
    current_bbox: List[int] = field(default_factory=list)
    phone_use_duration: float = 0.0
    productivity_score: float = 100.0
    recognition_confidence: float = 0.0
    logged_left: bool = False
    
    # Pose estimation states
    pose_state: Optional[Dict[str, Any]] = None

    # Phone usage confirmation tracking
    phone_use_start: Optional[datetime] = None
    phone_confirmed_use_active: bool = False
    phone_use_history: List[Dict[str, Any]] = field(default_factory=list)

    # Accurate working time — accumulated only when status == "tracking".
    # This excludes all gaps where the person was lost/out-of-frame.
    accumulated_working_seconds: float = 0.0

    # Count of confirmed phone usage events (each start→end interval = 1 event)
    phone_use_count: int = 0

    # Cross-camera movement history: list of {cam_id, entry_time, exit_time}
    camera_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def working_duration(self) -> float:
        """Returns ACTUAL accumulated on-camera tracking time in seconds.
        
        Unlike (last_seen - first_seen), this value excludes all gaps where
        the session was in "lost" status — lunch breaks, walking behind
        racks, RTSP glitches, etc.
        """
        return self.accumulated_working_seconds


class GlobalSessionManager:
    """Manages active global employee sessions across multiple cameras, with cross-camera stitching support."""

    
    
    def __init__(self, lost_timeout_seconds: int = 30) -> None:
        self.sessions: Dict[str, GlobalSession] = {}  # session_id -> GlobalSession
        self.lost_timeout = timedelta(seconds=lost_timeout_seconds)


    def get_session_by_track(self, camera_id: str, track_id: int) -> Optional[GlobalSession]:
        """Find an active session associated with a specific camera and track ID."""
        for session in self.sessions.values():
            if session.status != "exited" and camera_id in session.visible_cameras:
                if session.visible_cameras[camera_id].track_id == track_id:
                    return session
        return None

    def create_session(
        self,
        employee_id: str,
        employee_name: str,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        confidence: float,
        reid_features: Optional[np.ndarray] = None,
        reid_hist: Optional[np.ndarray] = None,
    ) -> GlobalSession:
        """Create a new global session or reactivate an existing one.

        After creation, we attempt to stitch this session with any other active
        sessions that belong to the same employee on a different camera.
        """
        if reid_features is None and reid_hist is not None:
            reid_features = reid_hist

        # Check for an existing non-exited session for this employee to prevent duplicates
        for session in self.sessions.values():
            if session.employee_id == employee_id and session.status != "exited":
                # Reactivate existing session
                session.status = "tracking"
                self.bind_camera_track(session, camera_id, track_id, bbox, timestamp, reid_features)
                return session

        # Create a new session if none exists
        session_id = f"SESS_{employee_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        session = GlobalSession(
            session_id=session_id,
            employee_id=employee_id,
            employee_name=employee_name,
            status="tracking",
            first_seen=timestamp,
            last_seen=timestamp,
            reid_features=reid_features,
            reid_hist=reid_features if reid_features is not None and reid_features.ndim > 1 else None,
            visible_cameras={camera_id: CameraTrackState(track_id, bbox, timestamp)},
            current_track_id=track_id,
            current_bbox=list(bbox),
            phone_use_duration=0.0,
            productivity_score=100.0,
            recognition_confidence=confidence,
            logged_left=False,
        )
        self.sessions[session_id] = session
        # Attempt cross‑camera stitching with any other session of the same employee
        self._stitch_cross_camera(session)
        return session

    def update_track(
        self,
        session: GlobalSession,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        phone_dets: List[Any],
        pose_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update an existing session's tracking state on a camera.

        After the normal update we also invoke cross‑camera stitching to merge
        any other sessions that share the same employee ID.
        """
        session.last_seen = timestamp
        session.status = "tracking"
        session.current_track_id = track_id
        session.current_bbox = list(bbox)
        session.pose_state = pose_state

        # Ensure camera track state exists
        if camera_id not in session.visible_cameras:
            session.visible_cameras[camera_id] = CameraTrackState(track_id, bbox, timestamp)
        
        cam_state = session.visible_cameras[camera_id]
        prev_seen = cam_state.last_seen
        cam_state.bbox = list(bbox)
        cam_state.last_seen = timestamp
        cam_state.pose_state = pose_state

        # ── Accumulate REAL working time (excludes gaps) ─────────────────────
        if prev_seen is not None and session.status == "tracking":
            dt = (timestamp - prev_seen).total_seconds()
            if 0.0 < dt < 10.0:
                session.accumulated_working_seconds += dt

        # ── Phone-Hand Proximity Check ──────────────────────────────────
        phone_used = False
        px1, py1, px2, py2 = bbox
        pw = px2 - px1
        ph = py2 - py1

        # Use dynamic threshold for proximity: 15% of the person's bounding box size
        proximity_threshold = 0.15 * max(pw, ph)

        for phone in phone_dets:
            ph_x1, ph_y1, ph_x2, ph_y2 = phone.bbox
            ph_cx = (ph_x1 + ph_x2) / 2.0
            ph_cy = (ph_y1 + ph_y2) / 2.0
            
            # 1. Spatial overlap: center of phone must be inside the person's bounding box
            if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                # 2. Hand proximity: check distance to left/right hands from MediaPipe Pose
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
                
                # If we have pose landmarks, enforce proximity. If not, fallback to spatial overlap.
                if pose_state and pose_state.get("landmarks"):
                    if has_hand_proximity:
                        phone_used = True
                        break
                else:
                    phone_used = True
                    break

        cam_state.phone_use_detected = phone_used

        # ── Timer & Confirmation logic ──────────────────────────────────
        if phone_used:
            if session.phone_use_start is None:
                session.phone_use_start = timestamp
            else:
                overlap_time = (timestamp - session.phone_use_start).total_seconds()
                if overlap_time >= settings.PHONE_USAGE_CONFIRM_SECONDS:
                    # Confirmed phone usage active! Accumulate duration
                    if prev_seen is not None:
                        dt = (timestamp - prev_seen).total_seconds()
                        if 0.0 < dt < 5.0:  # Ignore anomalies
                            session.phone_use_duration += dt
                    session.phone_confirmed_use_active = True
        else:
            # Overlap ended. If usage was active, close the interval
            if session.phone_use_start is not None:
                duration = (timestamp - session.phone_use_start).total_seconds()
                if duration >= settings.PHONE_USAGE_CONFIRM_SECONDS:
                    session.phone_use_history.append({
                        "start": session.phone_use_start,
                        "end": timestamp,
                        "duration": duration
                    })
                session.phone_use_start = None
                session.phone_confirmed_use_active = False

        # Update global productivity score
        total_duration = session.working_duration
        if total_duration > 0.0:
            non_phone_time = total_duration - session.phone_use_duration
            session.productivity_score = max(0.0, min(100.0, 100.0 * (non_phone_time / total_duration)))
        else:
            session.productivity_score = 100.0
            
        # After all updates, attempt to stitch with other sessions of the same employee.
        self._stitch_cross_camera(session)

    def _stitch_cross_camera(self, primary: GlobalSession) -> None:
        """Merge *primary* with any other session that belongs to the same
        employee but appears on a different camera.

        The merge consolidates ``visible_cameras`` and ``camera_history`` and
        updates ``first_seen``/``last_seen`` to reflect the combined timeframe.
        After merging the secondary session is removed from ``self.sessions``.
        """
        for sid, other in list(self.sessions.items()):
            if other is primary:
                continue
            if other.employee_id == primary.employee_id and other.status != "exited":
                # Merge visible cameras (do not overwrite existing entries)
                for cam_id, cam_state in other.visible_cameras.items():
                    if cam_id not in primary.visible_cameras:
                        primary.visible_cameras[cam_id] = cam_state
                # Merge camera history
                primary.camera_history.extend(other.camera_history)
                # Update timestamps
                if other.first_seen < primary.first_seen:
                    primary.first_seen = other.first_seen
                if other.last_seen > primary.last_seen:
                    primary.last_seen = other.last_seen
                # Accumulate working seconds, phone usage, etc.
                primary.accumulated_working_seconds += other.accumulated_working_seconds
                primary.phone_use_duration += other.phone_use_duration
                primary.phone_use_history.extend(other.phone_use_history)
                primary.phone_use_count += other.phone_use_count
                # Remove the duplicated session
                del self.sessions[sid]
                logger.info(
                    "[GlobalSessionManager] Merged session %s into %s for employee %s",
                    sid,
                    primary.session_id,
                    primary.employee_id,
                )
                # Only one merge expected per call
                break

    def bind_camera_track(
        self,
        session: GlobalSession,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        reid_features: Optional[np.ndarray] = None,
        reid_hist: Optional[np.ndarray] = None,
    ) -> None:
        """Bind/reconnect a camera track ID to a global session.

        This method also starts a new entry in ``camera_history`` for the given
        camera.  No stitching is performed here because the session already
        belongs to the correct employee.
        """
        if reid_features is None and reid_hist is not None:
            reid_features = reid_hist

        session.last_seen = timestamp
        session.current_track_id = track_id
        session.current_bbox = list(bbox)
        if reid_features is not None:
            session.reid_features = reid_features
            if reid_features.ndim > 1:
                session.reid_hist = reid_features

        session.visible_cameras[camera_id] = CameraTrackState(
            track_id=track_id,
            bbox=list(bbox),
            last_seen=timestamp
        )

        # Open a new camera_history entry for this camera appearance
        session.camera_history.append({
            "cam_id": camera_id,
            "entry_time": timestamp,
            "exit_time": None,
        })

    def handle_lost_track(self, camera_id: str, track_id: int, timestamp: datetime) -> None:
        """Mark a track as lost on a specific camera.

        The method closes the ``camera_history`` entry for the camera and, if the
        employee is no longer visible on any camera, marks the session as ``lost``.
        """
        for session in self.sessions.values():
            if session.status != "exited" and camera_id in session.visible_cameras:
                if session.visible_cameras[camera_id].track_id == track_id:
                    # Close the open camera_history entry for this camera
                    for entry in reversed(session.camera_history):
                        if entry["cam_id"] == camera_id and entry["exit_time"] is None:
                            entry["exit_time"] = timestamp
                            break
                    # Remove from active visible cameras
                    del session.visible_cameras[camera_id]
                    # If not visible on any camera, set global status to lost
                    if not session.visible_cameras:
                        session.status = "lost"
                        session.last_seen = timestamp
                    break

    def process_timeouts(self, timestamp: datetime) -> List[GlobalSession]:
        """Find and return sessions that have been lost for longer than the timeout."""
        exited = []
        for session in list(self.sessions.values()):
            if session.status == "lost":
                if timestamp - session.last_seen > self.lost_timeout:
                    # Finalise phone usage if active … (unchanged) …
                    session.status = "exited"
                    exited.append(session)
        return exited
        """Find and return sessions that have been lost for longer than the timeout."""
        exited = []
        for session in list(self.sessions.values()):
            if session.status == "lost":
                if timestamp - session.last_seen > self.lost_timeout:
                    # If phone usage was active during timeout, finalize it
                    if session.phone_use_start is not None:
                        duration = (session.last_seen - session.phone_use_start).total_seconds()
                        if duration >= settings.PHONE_USAGE_CONFIRM_SECONDS:
                            session.phone_use_history.append({
                                "start": session.phone_use_start,
                                "end": session.last_seen,
                                "duration": duration
                            })
                            session.phone_use_count += 1
                        session.phone_use_start = None
                        session.phone_confirmed_use_active = False
                    
                    session.status = "exited"
                    exited.append(session)
        return exited

