# person_management/person_manager.py
"""PersonManager maintains state for each tracked person.

It handles track state transitions, face recognition status, attendance sessions,
mobile phone usage monitoring, and employee productivity scoring.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

@dataclass
class PersonState:
    """State information for a currently tracked person."""
    track_id: int
    bbox: List[int]
    confidence: float
    first_seen: datetime
    last_seen: datetime
    camera_id: str
    status: Literal["tracking", "lost", "exited"] = "tracking"
    frame_count: int = 0
    recognition_status: Literal["unknown", "pending", "identified"] = "unknown"
    employee_id: str | None = None
    employee_name: str = "Unknown"
    recognition_confidence: float = 0.0
    session_start_time: Optional[datetime] = None
    
    # Persistent recognition retry and appearance ReID
    last_recognition_attempt: Optional[datetime] = None
    reid_hist: Optional[Any] = None
    
    # Enriched state fields for employee productivity monitoring
    phone_use_detected: bool = False
    phone_use_duration: float = 0.0          # total seconds phone was used
    total_tracked_duration: float = 0.0      # total seconds tracked
    productivity_score: float = 100.0        # percentage (0.0 to 100.0)
    prev_seen: Optional[datetime] = None


class PersonManager:
    """Manage a collection of PersonState objects."""
    LOST_TIMEOUT = timedelta(seconds=2)

    def __init__(self) -> None:
        self._persons: Dict[int, PersonState] = {}

    def process_tracks(self, tracks: List[object], timestamp: datetime) -> List[PersonState]:
        """Backward-compatible wrapper mapping to process_tracks_with_phones."""
        return self.process_tracks_with_phones("default", timestamp, tracks, [])

    def update(self, camera_id: str, timestamp: datetime, tracks: List[object]) -> List[PersonState]:
        """Backward-compatible update mapping to process_tracks_with_phones."""
        return self.process_tracks_with_phones(camera_id, timestamp, tracks, [])

    def process_tracks_with_phones(
        self,
        camera_id: str,
        timestamp: datetime,
        tracks: List[object],
        phone_detections: List[object]
    ) -> List[PersonState]:
        """Update internal track states and perform spatial phone overlap analysis."""
        current_ids = {t.track_id for t in tracks}

        # 1. Update tracks
        for trk in tracks:
            if trk.track_id in self._persons:
                p = self._persons[trk.track_id]
                p.prev_seen = p.last_seen
                p.bbox = trk.bbox
                p.confidence = trk.confidence
                p.last_seen = timestamp
                p.frame_count += 1
                p.status = "tracking"
                p.camera_id = camera_id
            else:
                self._persons[trk.track_id] = PersonState(
                    track_id=trk.track_id,
                    bbox=trk.bbox,
                    confidence=trk.confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    camera_id=camera_id,
                    frame_count=1,
                    recognition_status="unknown"
                )

        # 2. Handle missing IDs and clean up stale exited tracks
        for pid, person in list(self._persons.items()):
            if pid not in current_ids:
                if timestamp - person.last_seen > self.LOST_TIMEOUT:
                    person.status = "exited"
                    # Clean up track from registry after 5 minutes of exit to prevent memory leaks
                    if timestamp - person.last_seen > timedelta(minutes=5):
                        del self._persons[pid]
                else:
                    person.status = "lost"

        # 3. For all currently tracking identified employees, check phone usage and calculate metrics
        for pid, person in self._persons.items():
            if person.status == "tracking":
                person.total_tracked_duration = (person.last_seen - person.first_seen).total_seconds()
                
                # Spatial phone usage check and productivity score calculations only apply to identified employees
                if person.recognition_status == "identified":
                    phone_used = False
                    px1, py1, px2, py2 = person.bbox
                    
                    for phone in phone_detections:
                        ph_x1, ph_y1, ph_x2, ph_y2 = phone.bbox
                        # A cell phone is used by a person if its center resides inside the person's bbox
                        ph_cx = (ph_x1 + ph_x2) / 2.0
                        ph_cy = (ph_y1 + ph_y2) / 2.0
                        
                        if px1 <= ph_cx <= px2 and py1 <= ph_cy <= py2:
                            phone_used = True
                            break
                    
                    person.phone_use_detected = phone_used
                    
                    # Accumulate phone usage duration
                    if phone_used and person.prev_seen is not None:
                        dt = (person.last_seen - person.prev_seen).total_seconds()
                        if 0.0 < dt < 5.0:  # ignore huge anomalies
                            person.phone_use_duration += dt

                    # Calculate productivity score
                    if person.total_tracked_duration > 0.0:
                        non_phone_time = person.total_tracked_duration - person.phone_use_duration
                        person.productivity_score = max(0.0, min(100.0, 100.0 * (non_phone_time / person.total_tracked_duration)))
                    else:
                        person.productivity_score = 100.0
                else:
                    # Non-identified or unknown persons do not get phone or productivity monitoring
                    person.phone_use_detected = False
                    person.productivity_score = 100.0

        # Return active persons (tracking or lost)
        return [p for p in self._persons.values() if p.status != "exited"]
