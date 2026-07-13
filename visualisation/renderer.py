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
            if p.status != "tracking":
                continue
                
            x1, y1, x2, y2 = p.bbox
            
            # Determine color and text depending on recognition status
            lines = []
            if p.recognition_status == "identified":
                if p.phone_use_detected:
                    color = (40, 40, 255)  # Red warning color if using phone
                else:
                    color = (40, 220, 40)  # Green for identified
                
                # Compute session duration
                session_start = getattr(p, "session_start_time", None)
                if session_start is not None:
                    duration_sec = (p.last_seen - session_start).total_seconds()
                    m, s = divmod(int(duration_sec), 60)
                    h, m = divmod(m, 60)
                    session_time = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    session_time = "00:00:00"

                lines = [
                    f"Employee: {p.employee_name}",
                    f"ID: {p.employee_id} | Track: {p.track_id}",
                    f"Match: {getattr(p, 'recognition_confidence', 0.0):.1f}% | Status: Active",
                    f"Prod Timer: {session_time}",
                    f"Prod Score: {p.productivity_score:.1f}%"
                ]
                if p.phone_use_detected:
                    lines.append(f"Phone Use: {p.phone_use_duration:.1f}s [WARNING]")
            else:
                color = (0, 140, 255)  # Orange for unknown
                lines = [
                    f"Track: {p.track_id}",
                    "Status: Unknown"
                ]
            
            thickness = 3 if (p.recognition_status == "identified" and p.phone_use_detected) else 2
            
            # Draw person bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw multi-line text block below the bounding box
            line_height = 20
            block_height = len(lines) * line_height + 10
            
            # Find the max width for the background block
            max_width = 0
            for line in lines:
                size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                if size[0] > max_width:
                    max_width = size[0]
                    
            cv2.rectangle(overlay, (x1, y2), (x1 + max_width + 10, y2 + block_height), (20, 20, 20), -1)
            
            # Draw each line
            for i, line in enumerate(lines):
                text_color = (200, 255, 200) if p.recognition_status == "identified" else (255, 255, 255)
                if "[WARNING]" in line:
                    text_color = (100, 100, 255)
                
                cv2.putText(
                    overlay,
                    line,
                    (x1 + 5, y2 + 15 + (i * line_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    1,
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
