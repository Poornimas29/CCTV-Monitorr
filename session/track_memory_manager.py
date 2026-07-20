# session/track_memory_manager.py
"""TrackMemoryManager manages the TrackMemory dictionary and track timeouts.

RC-6 fix: Replaced consecutive_count + last_matched_employee_id with vote_buffer
(collections.deque, maxlen=5). The identity_manager uses this rolling window to
lock identity when 3 of the last 5 recognition results agree — one bad frame no
longer resets all recognition progress.

RC-9 fix: All print() calls replaced with logger.info/debug. Console print in the
hot path (called every frame per track) is the single largest source of FPS loss
on Windows due to GIL acquisition + synchronous stdout I/O.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import config.settings as settings

logger = logging.getLogger(__name__)


class TrackMemoryManager:
    """Manages local camera TrackMemory state, logging, and track lifecycle transitions."""

    def __init__(self) -> None:
        # Dictionary mapping (camera_id, track_id) -> Track State Dict
        self.tracks: Dict[tuple, Dict[str, Any]] = {}

    def create_track(
        self, camera_id: str, track_id: int, bbox: List[int], timestamp: datetime
    ) -> Dict[str, Any]:
        """Creates a new track memory entry for a newly detected person."""
        key = (camera_id, track_id)
        track_state = {
            "track_id": track_id,
            "employee_id": None,
            "employee_name": "Unknown",
            "recognition_status": "unknown",
            "locked_status": False,
            "embedding_history": [],
            "recognition_count": 0,
            # ── Voting window (RC-6) ──────────────────────────────────────────
            # Replaces consecutive_count + last_matched_employee_id.
            # Stores the last 5 recognition outcomes (employee_id or None).
            # Identity is locked when 3+ of the last 5 agree on the same ID.
            "vote_buffer": deque(maxlen=5),
            # ─────────────────────────────────────────────────────────────────
            "current_bbox": list(bbox),
            "bbox": list(bbox),
            "entry_time": timestamp,
            "last_seen_time": timestamp,
            "prev_seen_time": timestamp,
            "exit_time": None,
            "camera_id": camera_id,
            "recognition_confidence": 0.0,
            "track_status": "tracking",
            "last_recognition_attempt": None,
            "reid_feature_history": [],
            # ── Phone Usage Tracking for Unknown Tracks ───────────────────────
            "phone_use_duration": 0.0,
            "phone_use_count": 0,
            "phone_use_start": None,
            "phone_confirmed_use_active": False,
        }
        self.tracks[key] = track_state
        # Significant event: log at INFO, not print
        logger.info(
            "[TrackMemory] Created — Track %d on Camera %s", track_id, camera_id
        )
        return track_state

    def update_track(
        self, camera_id: str, track_id: int, bbox: List[int], timestamp: datetime
    ) -> Dict[str, Any]:
        """Updates the bounding box and last seen time for an active track."""
        key = (camera_id, track_id)
        if key not in self.tracks:
            return self.create_track(camera_id, track_id, bbox, timestamp)

        track = self.tracks[key]
        track["prev_seen_time"] = track.get("last_seen_time", timestamp)
        track["current_bbox"] = list(bbox)
        track["bbox"] = list(bbox)
        track["last_seen_time"] = timestamp

        if track["track_status"] == "lost":
            track["track_status"] = "tracking"
            logger.info(
                "[TrackMemory] Recovered — Track %d on Camera %s (Employee: %s)",
                track_id,
                camera_id,
                track.get("employee_id") or "Unknown",
            )

        logger.debug(
            "[TrackMemory] Updated — Track %d on Camera %s", track_id, camera_id
        )
        return track

    def get_track(self, camera_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a track by camera ID and track ID."""
        return self.tracks.get((camera_id, track_id))

    def mark_lost(
        self, camera_id: str, track_id: int, timestamp: datetime
    ) -> None:
        """Transitions a track to lost status."""
        key = (camera_id, track_id)
        if key in self.tracks:
            track = self.tracks[key]
            if track["track_status"] == "tracking":
                track["track_status"] = "lost"
                track["last_seen_time"] = timestamp
                logger.info(
                    "[TrackMemory] Lost — Track %d on Camera %s (Employee: %s)",
                    track_id,
                    camera_id,
                    track.get("employee_id") or "Unknown",
                )

    def process_timeouts(
        self, timestamp: datetime, timeout_seconds: float = None
    ) -> List[Dict[str, Any]]:
        """Identifies tracks that have been lost longer than timeout and marks them exited."""
        timeout_val = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.TRACK_TIMEOUT
        )
        timeout_delta = timedelta(seconds=timeout_val)
        exited_tracks = []

        for key, track in list(self.tracks.items()):
            if track["track_status"] == "lost":
                if timestamp - track["last_seen_time"] > timeout_delta:
                    track["track_status"] = "exited"
                    track["exit_time"] = track["last_seen_time"]
                    exited_tracks.append(track)
                    logger.info(
                        "[TrackMemory] Exited — Track %d on Camera %s (Employee: %s)",
                        track["track_id"],
                        track["camera_id"],
                        track.get("employee_id") or "Unknown",
                    )
        return exited_tracks
