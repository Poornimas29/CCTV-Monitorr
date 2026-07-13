# visualisation/renderer.py
"""Renderer draws bounding boxes, track IDs, FPS and person count.

The visual style uses vibrant colors per track ID, a semi‑transparent FPS
overlay, and a clean dark‑theme background suitable for premium UI.
"""

import cv2
import random
from typing import List
from person_management.person_manager import PersonState

class Renderer:
    """Render visual annotations on a frame.

    The class caches a random colour per ``track_id`` to keep colours stable
    across frames.  Colours are chosen from a bright palette for visibility on
    any background.
    """

    def __init__(self):
        self._colors: dict[int, tuple[int, int, int]] = {}

    def _color_for_id(self, track_id: int) -> tuple[int, int, int]:
        if track_id not in self._colors:
            # Generate a bright colour (avoid very dark shades).
            self._colors[track_id] = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255),
            )
        return self._colors[track_id]

    def draw(self, frame, persons: List[PersonState], fps: float = 0.0):
        """Draw bounding boxes and overlay information on *frame*.

        Parameters
        ----------
        frame: np.ndarray
            BGR image to annotate.
        persons: List[PersonState]
            Current persons to visualise.
        fps: float, optional
            Frames‑per‑second value for the overlay.
        """
        overlay = frame.copy()
        for p in persons:
            color = self._color_for_id(p.track_id)
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"ID:{p.track_id}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        # FPS and total count overlay (semi‑transparent black bar).
        h, w = frame.shape[:2]
        bar_h = 40
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            overlay,
            f"Persons: {len(persons)}",
            (w - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
        )
        # Blend overlay with original frame for smooth appearance.
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame
