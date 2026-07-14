# person_management/employee_session_manager.py
"""EmployeeSessionManager manages employee identity sessions decoupled from raw tracker track IDs.

It maintains tracking state, session recovery, appearance ReID mapping, phone usage warnings,
and productivity metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional
import numpy as np

@dataclass
class EmployeeSession:
    """Represents a persistent tracking session of an identified employee."""
    session_id: str
    employee_id: str
    employee_name: str
    track_id: int
    camera_id: str
    bbox: List[int]
    first_seen: datetime
    last_seen: datetime
    status: Literal["tracking", "lost", "exited"] = "tracking"
    recognition_confidence: float = 0.0
    reid_hist: Optional[Any] = None
    
    # Enriched analytics metrics
    phone_use_detected: bool = False
    phone_use_duration: float = 0.0
    total_tracked_duration: float = 0.0
    productivity_score: float = 100.0
    prev_seen: Optional[datetime] = None
    prev_bbox: List[int] = None


class EmployeeSessionManager:
    """Manages active employee sessions and unrecognized tracks for a camera stream."""
    
    def __init__(self, lost_timeout_seconds: int = 30) -> None:
        self.sessions: Dict[str, EmployeeSession] = {}  # session_id -> EmployeeSession
        self.lost_timeout = timedelta(seconds=lost_timeout_seconds)

    def get_session_by_track_id(self, track_id: int) -> Optional[EmployeeSession]:
        """Find an active or lost employee session associated with a track ID."""
        for session in self.sessions.values():
            if session.track_id == track_id and session.status != "exited":
                return session
        return None

    def create_session(
        self,
        session_id: str,
        employee_id: str,
        employee_name: str,
        track_id: int,
        camera_id: str,
        bbox: List[int],
        timestamp: datetime,
        confidence: float,
        reid_hist: Optional[np.ndarray] = None
    ) -> EmployeeSession:
        """Create a new employee session."""
        session = EmployeeSession(
            session_id=session_id,
            employee_id=employee_id,
            employee_name=employee_name,
            track_id=track_id,
            camera_id=camera_id,
            bbox=bbox,
            first_seen=timestamp,
            last_seen=timestamp,
            recognition_confidence=confidence,
            reid_hist=reid_hist,
            prev_bbox=list(bbox)
        )
        self.sessions[session_id] = session
        return session

    def update_session(self, session: EmployeeSession, bbox: List[int], timestamp: datetime) -> None:
        """Update an existing session with new coordinates and timestamps."""
        session.prev_bbox = list(session.bbox)
        session.prev_seen = session.last_seen
        session.bbox = bbox
        session.last_seen = timestamp
        session.status = "tracking"
        session.total_tracked_duration = (session.last_seen - session.first_seen).total_seconds()

    def update_metrics(self, session: EmployeeSession, phone_detections: List[Any]) -> None:
        """Calculate mobile phone overlap and productivity metrics."""
        phone_used = False
        px1, py1, px2, py2 = session.bbox

        for phone in phone_detections:
            ph_x1, ph_y1, ph_x2, ph_y2 = phone.bbox
            ph_cx = (ph_x1 + ph_x2) / 2.0
            ph_cy = (ph_y1 + ph_y2) / 2.0

            if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                phone_used = True
                break

        session.phone_use_detected = phone_used

        if phone_used and session.prev_seen is not None:
            dt = (session.last_seen - session.prev_seen).total_seconds()
            if 0.0 < dt < 5.0:  # Ignore anomalies
                session.phone_use_duration += dt

        if session.total_tracked_duration > 0.0:
            non_phone_time = session.total_tracked_duration - session.phone_use_duration
            session.productivity_score = max(0.0, min(100.0, 100.0 * (non_phone_time / session.total_tracked_duration)))
        else:
            session.productivity_score = 100.0
