# tests/test_global_session_manager.py
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session.global_session_manager import GlobalSessionManager, GlobalSession, CameraTrackState

class TestGlobalSessionManager(unittest.TestCase):
    def setUp(self) -> None:
        import config.settings as settings
        settings.PHONE_USAGE_CONFIRM_SECONDS = 0.0
        self.manager = GlobalSessionManager(lost_timeout_seconds=30)
        self.base_time = datetime(2026, 7, 14, 18, 0, 0)

    def test_create_session(self) -> None:
        # Create a new session
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        self.assertIsNotNone(session)
        self.assertEqual(session.employee_id, "EMP001")
        self.assertEqual(session.employee_name, "Arun")
        self.assertEqual(session.status, "tracking")
        self.assertEqual(session.current_track_id, 1)
        self.assertEqual(session.current_bbox, [100, 100, 200, 200])
        self.assertIn("CAM001", session.visible_cameras)
        self.assertEqual(session.visible_cameras["CAM001"].track_id, 1)
        self.assertEqual(session.working_duration, 0.0)

    def test_create_session_existing_reuses(self) -> None:
        # Create initial session
        s1 = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        # Attempt to create session for same employee ID on another camera
        s2 = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM002",
            track_id=2,
            bbox=[150, 150, 250, 250],
            timestamp=self.base_time + timedelta(seconds=5),
            confidence=90.0,
            reid_hist=None
        )

        self.assertEqual(s1.session_id, s2.session_id)
        self.assertEqual(s2.status, "tracking")
        self.assertIn("CAM001", s2.visible_cameras)
        self.assertIn("CAM002", s2.visible_cameras)
        self.assertEqual(s2.visible_cameras["CAM002"].track_id, 2)
        self.assertEqual(s2.working_duration, 5.0)

    def test_get_session_by_track(self) -> None:
        s1 = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        found = self.manager.get_session_by_track("CAM001", 1)
        self.assertEqual(found, s1)

        not_found = self.manager.get_session_by_track("CAM001", 2)
        self.assertIsNone(not_found)

        not_found_cam = self.manager.get_session_by_track("CAM002", 1)
        self.assertIsNone(not_found_cam)

    def test_update_track_and_metrics(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        # Mock a phone detection overlapping with the bbox [100, 100, 200, 200]
        # Center of phone bbox will be at (150, 150) which is inside
        mock_phone = MagicMock()
        mock_phone.bbox = [140, 140, 160, 160]

        # Update track at base_time to start phone overlap interval
        self.manager.update_track(
            session=session,
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            phone_dets=[mock_phone]
        )

        # Update track with phone use at t=3s
        self.manager.update_track(
            session=session,
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time + timedelta(seconds=3),
            phone_dets=[mock_phone]
        )

        self.assertTrue(session.visible_cameras["CAM001"].phone_use_detected)
        self.assertEqual(session.phone_use_duration, 3.0)
        self.assertEqual(session.working_duration, 3.0)
        self.assertEqual(session.productivity_score, 0.0)

    def test_handle_lost_track(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        # Mark track lost on CAM001
        self.manager.handle_lost_track("CAM001", 1, self.base_time + timedelta(seconds=5))

        self.assertNotIn("CAM001", session.visible_cameras)
        self.assertEqual(session.status, "lost")
        self.assertEqual(session.last_seen, self.base_time + timedelta(seconds=5))

    def test_process_timeouts(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0,
            reid_hist=None
        )

        # Mark track lost
        self.manager.handle_lost_track("CAM001", 1, self.base_time)

        # Check timeouts before timeout interval
        exited = self.manager.process_timeouts(self.base_time + timedelta(seconds=10))
        self.assertEqual(len(exited), 0)
        self.assertEqual(session.status, "lost")

        # Check timeouts after timeout interval
        exited = self.manager.process_timeouts(self.base_time + timedelta(seconds=35))
        self.assertEqual(len(exited), 1)
        self.assertEqual(exited[0], session)
        self.assertEqual(session.status, "exited")

if __name__ == "__main__":
    unittest.main()
