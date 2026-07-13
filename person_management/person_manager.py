# person_management/person_manager.py
"""PersonManager maintains state for each tracked person.

It stores information required for the Phase 1 output payload and handles
lifecycle events (enter, continue, lost, exit).  The implementation is
lightweight and CPU‑friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Literal


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
    employee_id: int | None = None
    employee_name: str = "Unknown"


class PersonManager:
    """Manage a collection of PersonState objects.

    The manager is deliberately simple: it updates or creates entries for each
    incoming track and marks missing tracks as ``lost``.  If a track remains
    missing for longer than :attr:`LOST_TIMEOUT` it is marked ``exited`` and
    eventually pruned.
    """

    LOST_TIMEOUT = timedelta(seconds=2)

    def __init__(self) -> None:
        self._persons: Dict[int, PersonState] = {}

    def update(self, camera_id: str, timestamp: datetime, tracks: List[object]) -> List[PersonState]:
        """Update internal state with the latest tracks.

        Parameters
        ----------
        camera_id: str
            Identifier of the camera that produced the tracks.
        timestamp: datetime
            Timestamp of the current frame.
        tracks: List[object]
            List of track objects that have ``track_id``, ``bbox`` and ``confidence``
            attributes (as produced by ``tracking.tracker.Tracker``).
        """
        current_ids = {t.track_id for t in tracks}

        # Add new tracks or update existing ones
        for trk in tracks:
            if trk.track_id in self._persons:
                p = self._persons[trk.track_id]
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
                )

        # Handle missing IDs
        for pid, person in list(self._persons.items()):
            if pid not in current_ids:
                if timestamp - person.last_seen > self.LOST_TIMEOUT:
                    person.status = "exited"
                else:
                    person.status = "lost"

        # Return active persons (tracking or lost) – callers may filter further.
        return [p for p in self._persons.values() if p.status != "exited"]
