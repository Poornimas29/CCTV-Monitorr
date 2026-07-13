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
            x1, y1, x2, y2 = p.bbox
            
            # Determine color and text depending on recognition status
            if p.recognition_status == "identified":
                if p.phone_use_detected:
                    color = (40, 40, 255)  # Red warning color if using phone
                else:
                    color = (40, 220, 40)  # Green for identified
                name_label = f"[{p.track_id}] {p.employee_name}"
            else:
                color = (0, 140, 255)  # Orange for unknown
                name_label = f"[{p.track_id}] Unknown"
            
            thickness = 3 if (p.recognition_status == "identified" and p.phone_use_detected) else 2
            
            # Draw person bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw top text label background
            label_size = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + label_size[0] + 10, y1), color, -1)
            # Draw label text
            cv2.putText(
                overlay,
                name_label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            # Draw productivity info for identified employees
            if p.recognition_status == "identified":
                prod_text = f"Prod: {p.productivity_score:.1f}% | Phone: {p.phone_use_duration:.1f}s"
                if p.phone_use_detected:
                    prod_text += " [USING PHONE]"
                
                prod_size = cv2.getTextSize(prod_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                # Draw bottom text background
                cv2.rectangle(overlay, (x1, y2), (x1 + prod_size[0] + 10, y2 + 20), (20, 20, 20), -1)
                # Draw bottom text
                cv2.putText(
                    overlay,
                    prod_text,
                    (x1 + 5, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 255, 200) if not p.phone_use_detected else (150, 150, 255),
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
