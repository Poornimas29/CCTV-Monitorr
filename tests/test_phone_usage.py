# tests/test_phone_usage.py
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from session.global_session_manager import GlobalSessionManager, GlobalSession
import config.settings as settings

class TestPhoneUsage(unittest.TestCase):
    def setUp(self) -> None:
        settings.PHONE_USAGE_CONFIRM_SECONDS = 2.0
        self.manager = GlobalSessionManager()
        self.base_time = datetime(2026, 7, 15, 12, 0, 0)
        
    def test_phone_overlap_with_no_pose(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0
        )
        
        mock_phone = MagicMock()
        mock_phone.bbox = [140, 140, 160, 160] # Center is at (150, 150), inside person box
        
        # When pose_state is None, it should fall back to spatial overlap
        self.manager.update_track(
            session=session,
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time + timedelta(seconds=1),
            phone_dets=[mock_phone],
            pose_state=None
        )
        
        self.assertTrue(session.visible_cameras["CAM001"].phone_use_detected)
        self.assertEqual(session.phone_use_start, self.base_time + timedelta(seconds=1))
        # Not confirmed yet (requires CONFIRM_SECONDS, default 2.0s)
        self.assertFalse(session.phone_confirmed_use_active)

    def test_phone_overlap_with_pose_hand_proximity(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0
        )
        
        mock_phone = MagicMock()
        mock_phone.bbox = [140, 140, 160, 160] # Center is (150, 150)
        
        # Case 1: Pose is present but hand is far from phone center
        pose_state_far = {
            "landmarks": {15: {"x": 50, "y": 50}, 16: {"x": 50, "y": 50}}, # Left/Right hand
            "hands": {"left": (50, 50), "right": (50, 50)}
        }
        self.manager.update_track(
            session=session,
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time + timedelta(seconds=1),
            phone_dets=[mock_phone],
            pose_state=pose_state_far
        )
        # Should be ignored since it is far from hands
        self.assertFalse(session.visible_cameras["CAM001"].phone_use_detected)

        # Case 2: Pose is present and hand is close to phone center
        pose_state_close = {
            "landmarks": {15: {"x": 145, "y": 145}, 16: {"x": 50, "y": 50}},
            "hands": {"left": (145, 145), "right": (50, 50)}
        }
        self.manager.update_track(
            session=session,
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time + timedelta(seconds=2),
            phone_dets=[mock_phone],
            pose_state=pose_state_close
        )
        # Should be confirmed as detected phone use
        self.assertTrue(session.visible_cameras["CAM001"].phone_use_detected)

    def test_phone_use_confirmation_timer(self) -> None:
        session = self.manager.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=1,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=95.0
        )
        
        mock_phone = MagicMock()
        mock_phone.bbox = [140, 140, 160, 160]
        
        pose_state_close = {
            "landmarks": {15: {"x": 145, "y": 145}, 16: {"x": 50, "y": 50}},
            "hands": {"left": (145, 145), "right": (50, 50)}
        }
        
        # Frame 1: Phone overlap starts at t=0
        self.manager.update_track(session, "CAM001", 1, [100, 100, 200, 200], self.base_time, [mock_phone], pose_state_close)
        self.assertTrue(session.visible_cameras["CAM001"].phone_use_detected)
        self.assertFalse(session.phone_confirmed_use_active)
        self.assertEqual(session.phone_use_duration, 0.0)

        # Frame 2: Phone overlap at t=1.0s (under CONFIRM_SECONDS threshold 2.0s)
        self.manager.update_track(session, "CAM001", 1, [100, 100, 200, 200], self.base_time + timedelta(seconds=1), [mock_phone], pose_state_close)
        self.assertFalse(session.phone_confirmed_use_active)
        self.assertEqual(session.phone_use_duration, 0.0)

        # Frame 3: Phone overlap at t=2.0s (CONFIRM_SECONDS threshold reached)
        self.manager.update_track(session, "CAM001", 1, [100, 100, 200, 200], self.base_time + timedelta(seconds=2), [mock_phone], pose_state_close)
        self.assertTrue(session.phone_confirmed_use_active)
        # Duration should begin accumulating
        self.assertEqual(session.phone_use_duration, 1.0)

        # Frame 4: Phone overlap continues at t=3.0s
        self.manager.update_track(session, "CAM001", 1, [100, 100, 200, 200], self.base_time + timedelta(seconds=3), [mock_phone], pose_state_close)
        self.assertTrue(session.phone_confirmed_use_active)
        self.assertEqual(session.phone_use_duration, 2.0)

        # Frame 5: Phone overlap drops at t=4.0s
        self.manager.update_track(session, "CAM001", 1, [100, 100, 200, 200], self.base_time + timedelta(seconds=4), [], None)
        self.assertFalse(session.phone_confirmed_use_active)
        self.assertIsNone(session.phone_use_start)
        # Check that usage was recorded in history
        self.assertEqual(len(session.phone_use_history), 1)
        self.assertEqual(session.phone_use_history[0]["duration"], 4.0)

if __name__ == "__main__":
    unittest.main()
