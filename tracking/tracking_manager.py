# tracking/tracking_manager.py
"""TrackingManager manages ByteTrack-based tracking for a camera."""

from typing import List
from tracking.tracker import Tracker, Track
from detection.yolo26_detector import Detection
import config.settings as settings

class TrackingManager:
    """Manages a Tracker instance configured with track lost timeouts."""
    def __init__(self, track_timeout: float = None, fps: float = None) -> None:
        timeout = track_timeout if track_timeout is not None else settings.TRACK_TIMEOUT
        target_fps = fps if fps is not None else settings.TARGET_FPS
        self.max_lost = max(1, int(timeout * target_fps))
        self.tracker = Tracker(max_lost=self.max_lost)

    def update(self, detections: List[Detection]) -> List[Track]:
        """Updates the tracking state and returns the list of active tracks."""
        return self.tracker.update(detections)
