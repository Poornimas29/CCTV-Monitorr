# tracking/tracker.py
"""Tracker wrapper using ByteTrack (CPU implementation).

Exposes a Tracker class that receives detections, filters for person detections,
updates the ByteTracker, and returns STrack objects wrapped as Track containers.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from .byte_tracker import ByteTracker
from detection.yolo26_detector import Detection
import config.settings as settings

@dataclass
class Track:
    """Simple container representing a tracked object."""
    track_id: int
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float

class Tracker:
    """Wrap ByteTrack for person tracking.

    Parameters
    ----------
    max_lost: int, optional
        Number of consecutive frames a track can be missing before it is
        considered lost. Default matches the original ByteTrack paper.
    """
    def __init__(self, max_lost: int = 30):
        self.tracker = ByteTracker(max_lost=max_lost)

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections (persons only).

        Returns a list of :class:`Track` objects with persistent ``track_id``.
        """
        # Filter for person detections that meet the configurable confidence threshold
        person_dets = [
            d for d in detections 
            if d.class_id == 0 and d.confidence >= settings.CONF_PERSON
        ]
        
        if not person_dets:
            dets = np.empty((0, 5), dtype=np.float32)
        else:
            dets = np.array(
                [[*d.bbox, d.confidence] for d in person_dets],
                dtype=np.float32,
            )
        # ByteTracker expects detections as Nx5 array: x1, y1, x2, y2, score
        byte_tracks = self.tracker.update(dets)
        result: List[Track] = []
        for t in byte_tracks:
            bbox = [int(v) for v in t.tlbr]
            result.append(Track(track_id=int(t.track_id), bbox=bbox, confidence=float(t.score)))
        return result

