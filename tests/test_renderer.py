# tests/test_renderer.py
import os
import sys
import unittest
import numpy as np
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualisation.renderer import Renderer

class DummySession:
    def __init__(self, session_id, employee_id, employee_name, track_id, bbox, status, phone_use_detected=False, phone_use_duration=0.0, productivity_score=100.0, recognition_confidence=99.0):
        self.session_id = session_id
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.track_id = track_id
        self.bbox = bbox
        self.status = status
        self.first_seen = datetime(2026, 7, 14, 18, 0, 0)
        self.last_seen = datetime(2026, 7, 14, 18, 0, 10)
        self.phone_use_detected = phone_use_detected
        self.phone_use_duration = phone_use_duration
        self.productivity_score = productivity_score
        self.recognition_confidence = recognition_confidence
        self.is_recognized = (status == "tracking") and (employee_id is not None)

class TestRenderer(unittest.TestCase):
    def test_draw_with_recognized_and_unrecognized_tracks(self) -> None:
        renderer = Renderer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create one recognized session
        s1 = DummySession(
            session_id="SESS_EMP001",
            employee_id="EMP001",
            employee_name="Arun",
            track_id=1,
            bbox=[100, 100, 200, 300],
            status="tracking"
        )

        # Create one unrecognized track dict
        utrk = {
            "track_id": 2,
            "bbox": [300, 100, 400, 300],
            "face_bbox": [330, 110, 370, 150]
        }

        # Draw annotations
        annotated = renderer.draw(
            frame=frame,
            sessions=[s1],
            unrecognized_tracks=[utrk],
            fps=25.0
        )

        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (480, 640, 3))

if __name__ == "__main__":
    unittest.main()
