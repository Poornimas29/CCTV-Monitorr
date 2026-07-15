# visualisation/renderer.py
"""Renderer draws bounding boxes, track IDs, FPS and person count.

The visual style uses vibrant colors per track ID, a semi‑transparent FPS
overlay, and a clean dark‑theme background suitable for premium UI.
"""

import cv2
import random
from typing import List, Any

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

    def _draw_skeleton(self, frame, pose_state):
        if not pose_state or "skeleton" not in pose_state:
            return
        
        skeleton = pose_state["skeleton"]
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28)
        ]
        
        # Draw connections
        for p1, p2 in connections:
            if p1 < len(skeleton) and p2 < len(skeleton):
                x1, y1, vis1 = skeleton[p1]
                x2, y2, vis2 = skeleton[p2]
                if vis1 > 0.5 and vis2 > 0.5:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
        # Draw joint points
        for gx, gy, vis in skeleton:
            if vis > 0.5:
                cv2.circle(frame, (gx, gy), 4, (0, 255, 255), -1)

    def draw(
        self,
        frame,
        sessions: List[Any],
        unrecognized_tracks: List[dict],
        detections: List[Any] = None,
        fps: float = 0.0
    ):
        """Draw bounding boxes and overlay information on *frame*.

        Parameters
        ----------
        frame: np.ndarray
            BGR image to annotate.
        sessions: List[EmployeeSession]
            Active employee tracking sessions.
        unrecognized_tracks: List[dict]
            Active unrecognized tracking candidates.
        detections: List[Detection], optional
            All detections (including uniform, cap, phone).
        fps: float, optional
            Frames‑per‑second value for the overlay.
        """
        overlay = frame.copy()
        
        # 1. Draw skeletons for tracked people
        for s in sessions:
            if getattr(s, "pose_state", None) is not None:
                self._draw_skeleton(overlay, s.pose_state)
        for utrk in unrecognized_tracks:
            if utrk.get("pose_state") is not None:
                self._draw_skeleton(overlay, utrk["pose_state"])

        # 2. Draw Safety Cap, Uniform, and Mobile Phone detections
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                if det.class_id == 80:  # Uniform
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 1)
                    cv2.putText(
                        overlay, f"Uniform: {det.confidence:.2f}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 180, 0), 1, cv2.LINE_AA
                    )
                elif det.class_id == 81:  # Safety Cap
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    cv2.putText(
                        overlay, f"Safety Cap: {det.confidence:.2f}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 200, 255), 1, cv2.LINE_AA
                    )
                elif det.class_id == 67:  # Mobile Phone
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        overlay, f"Phone: {det.confidence:.2f}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1, cv2.LINE_AA
                    )

        # 3. Draw unrecognized tracks (ONLY face boxes, no body box)
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

        # 4. Draw active recognized employee sessions (ONLY body box, no face box)
        for s in sessions:
            is_recognized = getattr(s, "is_recognized", False)
            if not is_recognized or s.employee_id is None:
                continue
                
            x1, y1, x2, y2 = s.bbox
            phone_active = getattr(s, "phone_confirmed_use_active", False)
            if phone_active:
                color = (40, 40, 255)  # Red warning color if using phone
            else:
                color = (40, 220, 40)  # Green for identified
            
            # Compute session duration
            duration_sec = (s.last_seen - s.first_seen).total_seconds()
            m, s_val = divmod(int(duration_sec), 60)
            h, m = divmod(m, 60)
            session_time = f"{h:02d}:{m:02d}:{s_val:02d}"

            head_dir = "N/A"
            if getattr(s, "pose_state", None) is not None:
                head_dir = s.pose_state.get("head_direction", "Front")

            lines = [
                f"Employee: {s.employee_name}",
                f"ID: {s.employee_id} | Session: {s.session_id}",
                f"Track: {s.track_id} | Match: {getattr(s, 'recognition_confidence', 0.0):.1f}%",
                f"Head Dir: {head_dir}",
                f"Prod Timer: {session_time}",
                f"Start Time: {s.first_seen.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Prod Score: {s.productivity_score:.1f}%"
            ]
            if s.phone_use_duration > 0.0 or phone_active:
                warn_tag = " [WARNING]" if phone_active else ""
                lines.append(f"Phone Use: {s.phone_use_duration:.1f}s{warn_tag}")
        
            thickness = 3 if phone_active else 2
            
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

