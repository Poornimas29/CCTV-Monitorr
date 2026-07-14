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

    def draw(self, frame, sessions: List[Any], unrecognized_tracks: List[dict], fps: float = 0.0):
        """Draw bounding boxes and overlay information on *frame*.

        Parameters
        ----------
        frame: np.ndarray
            BGR image to annotate.
        sessions: List[EmployeeSession]
            Active employee tracking sessions.
        unrecognized_tracks: List[dict]
            Active unrecognized tracking candidates.
        fps: float, optional
            Frames‑per‑second value for the overlay.
        """
        overlay = frame.copy()
        
        # 1. Draw unrecognized tracks (ONLY face boxes, no body box)
        for utrk in unrecognized_tracks:
            face_bbox = utrk.get("face_bbox")
            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 140, 255), 2)
                
                # Draw "Unknown" label
                label = "Unknown"
                size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(overlay, (fx1, max(0, fy1 - 18)), (fx1 + size[0] + 10, fy1), (20, 20, 20), -1)
                cv2.putText(
                    overlay,
                    label,
                    (fx1 + 5, max(12, fy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw active recognized employee sessions (ONLY body box, no face box)
        for s in sessions:
            if s.status != "tracking":
                continue
                
            x1, y1, x2, y2 = s.bbox
            if s.phone_use_detected:
                color = (40, 40, 255)  # Red warning color if using phone
            else:
                color = (40, 220, 40)  # Green for identified
            
            # Compute session duration
            duration_sec = (s.last_seen - s.first_seen).total_seconds()
            m, s_val = divmod(int(duration_sec), 60)
            h, m = divmod(m, 60)
            session_time = f"{h:02d}:{m:02d}:{s_val:02d}"

            lines = [
                f"Employee: {s.employee_name}",
                f"ID: {s.employee_id} | Session: {s.session_id}",
                f"Track: {s.track_id} | Match: {getattr(s, 'recognition_confidence', 0.0):.1f}%",
                f"Prod Timer: {session_time}",
                f"Start Time: {s.first_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Prod Score: {s.productivity_score:.1f}%"
            ]
            if s.phone_use_detected:
                lines.append(f"Phone Use: {s.phone_use_duration:.1f}s [WARNING]")
        
            thickness = 3 if s.phone_use_detected else 2
            
            # Draw person body bbox
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
                text_color = (200, 255, 200)
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
            f"Persons: {len(sessions) + len(unrecognized_tracks)}",
            (w - 180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
        )
        # Blend overlay with original frame for smooth appearance.
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        return frame
