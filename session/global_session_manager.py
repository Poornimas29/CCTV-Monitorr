# session/global_session_manager.py
"""GlobalSessionManager manages multi-camera employee tracking sessions,
decoupled from raw camera-specific track IDs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

@dataclass
class CameraTrackState:
    """Represents the tracking state of a session on a specific camera."""
    track_id: int
    bbox: List[int]
    last_seen: datetime
    phone_use_detected: bool = False

@dataclass
class GlobalSession:
    """Represents a global tracking session of an identified employee across all cameras."""
    session_id: str
    employee_id: str
    employee_name: str
    status: str
    first_seen: datetime
    last_seen: datetime
    reid_hist: Optional[np.ndarray] = None
    visible_cameras: Dict[str, CameraTrackState] = field(default_factory=dict)
    current_track_id: int = -1
    current_bbox: List[int] = field(default_factory=list)
    phone_use_duration: float = 0.0
    productivity_score: float = 100.0
    recognition_confidence: float = 0.0
    logged_left: bool = False

    @property
    def working_duration(self) -> float:
        """Returns the total tracked duration in seconds."""
        if self.first_seen and self.last_seen:
            return (self.last_seen - self.first_seen).total_seconds()
        return 0.0


class GlobalSessionManager:
    """Manages active global employee sessions across multiple cameras."""
    
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
        reid_hist: Optional[np.ndarray] = None
    ) -> GlobalSession:
        """Create a new global session or reactivate/re-bind an existing non-exited session."""
        # Check for an existing non-exited session for this employee to prevent duplicates
        for session in self.sessions.values():
            if session.employee_id == employee_id and session.status != "exited":
                session.status = "tracking"
                self.bind_camera_track(session, camera_id, track_id, bbox, timestamp, reid_hist)
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
            reid_hist=reid_hist,
            visible_cameras={camera_id: CameraTrackState(track_id, bbox, timestamp)},
            current_track_id=track_id,
            current_bbox=list(bbox),
            phone_use_duration=0.0,
            productivity_score=100.0,
            recognition_confidence=confidence,
            logged_left=False
        )
        self.sessions[session_id] = session
        return session

    def update_track(
        self,
        session: GlobalSession,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        phone_dets: List[Any]
    ) -> None:
        """Update an existing session's tracking state on a camera, and update metrics."""
        session.last_seen = timestamp
        session.status = "tracking"
        session.current_track_id = track_id
        session.current_bbox = list(bbox)

        # Ensure camera track state exists
        if camera_id not in session.visible_cameras:
            session.visible_cameras[camera_id] = CameraTrackState(track_id, bbox, timestamp)
        
        cam_state = session.visible_cameras[camera_id]
        prev_seen = cam_state.last_seen
        cam_state.bbox = list(bbox)
        cam_state.last_seen = timestamp

        # Check for phone use overlap on this camera
        phone_used = False
        px1, py1, px2, py2 = bbox
        for phone in phone_dets:
            ph_x1, ph_y1, ph_x2, ph_y2 = phone.bbox
            ph_cx = (ph_x1 + ph_x2) / 2.0
            ph_cy = (ph_y1 + ph_y2) / 2.0
            if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                phone_used = True
                break

        cam_state.phone_use_detected = phone_used

        if phone_used and prev_seen is not None:
            dt = (timestamp - prev_seen).total_seconds()
            if 0.0 < dt < 5.0:  # Ignore anomalies
                session.phone_use_duration += dt

        # Update global productivity score
        total_duration = session.working_duration
        if total_duration > 0.0:
            non_phone_time = total_duration - session.phone_use_duration
            session.productivity_score = max(0.0, min(100.0, 100.0 * (non_phone_time / total_duration)))
        else:
            session.productivity_score = 100.0

    def bind_camera_track(
        self,
        session: GlobalSession,
        camera_id: str,
        track_id: int,
        bbox: List[int],
        timestamp: datetime,
        reid_hist: Optional[np.ndarray] = None
    ) -> None:
        """Bind/reconnect a camera track ID to a global session."""
        session.last_seen = timestamp
        session.current_track_id = track_id
        session.current_bbox = list(bbox)
        if reid_hist is not None:
            session.reid_hist = reid_hist

        session.visible_cameras[camera_id] = CameraTrackState(
            track_id=track_id,
            bbox=list(bbox),
            last_seen=timestamp
        )

    def handle_lost_track(self, camera_id: str, track_id: int, timestamp: datetime) -> None:
        """Mark a track as lost on a specific camera.

        If the session is no longer visible on any camera, its global status is set to 'lost'.
        """
        for session in self.sessions.values():
            if session.status != "exited" and camera_id in session.visible_cameras:
                if session.visible_cameras[camera_id].track_id == track_id:
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
                    session.status = "exited"
                    exited.append(session)
        return exited
