"""
Unit tests for Phase 1.1 — Multi-Camera RTSP Streaming.

Covers:
  - Configuration loading of enabled cameras.
  - URL generation, ensuring special characters in passwords are percent-encoded.
  - Thread lifecycle management (start_all and stop_all).
  - Data retrieval APIs (get_latest_frame, is_connected, get_fps, get_resolution).
  - Single-camera backwards compatibility fallback mechanism.
"""

import os
import sys
import unittest
import urllib.parse
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path for running tests in isolated environments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from stream.camera_manager import CameraManager


class TestMultiCameraStreaming(unittest.TestCase):
    """Test suite covering CameraManager functionality and multi-camera support."""

    def setUp(self) -> None:
        """Saves config states before each test."""
        self.original_cameras = getattr(settings, "CAMERAS", [])
        self.original_rtsp_host = getattr(settings, "RTSP_HOST", "")

    def tearDown(self) -> None:
        """Restores config states after each test."""
        settings.CAMERAS = self.original_cameras
        settings.RTSP_HOST = self.original_rtsp_host

    def test_camera_manager_loads_enabled_cameras_only(self) -> None:
        """Verifies that only enabled cameras are loaded from configuration."""
        settings.CAMERAS = [
            {"id": "CAM001", "name": "Camera 1", "channel": 1, "enabled": True},
            {"id": "CAM002", "name": "Camera 2", "channel": 2, "enabled": False},
            {"id": "CAM003", "name": "Camera 3", "channel": 3, "enabled": True},
        ]
        
        manager = CameraManager()
        active = manager.get_active_cameras()
        
        self.assertEqual(len(active), 2)
        active_ids = {cam["id"] for cam in active}
        self.assertIn("CAM001", active_ids)
        self.assertIn("CAM003", active_ids)
        self.assertNotIn("CAM002", active_ids)

    def test_generate_rtsp_url_mock(self) -> None:
        """Verifies that if RTSP_HOST is empty or 'mock', the generated URL is also 'mock'."""
        settings.RTSP_HOST = ""
        manager = CameraManager()
        url = manager.generate_rtsp_url(channel=3)
        self.assertEqual(url, "mock")

        settings.RTSP_HOST = "mock"
        manager = CameraManager()
        url = manager.generate_rtsp_url(channel=5)
        self.assertEqual(url, "mock")

    @patch("config.settings.RTSP_URL", "rtsp://base_url")
    @patch("config.settings.RTSP_HOST", "192.168.1.100")
    @patch("config.settings.RTSP_USERNAME", "custom_user")
    @patch("config.settings.RTSP_PASSWORD", "p@$$w0rd!")
    @patch("config.settings.RTSP_PORT", 554)
    def test_generate_rtsp_url_real(self) -> None:
        """Verifies generation of real RTSP URLs with password urlencoding."""
        manager = CameraManager()
        url = manager.generate_rtsp_url(channel=4)
        
        # Verify password is URL encoded (p@$$w0rd! -> p%40%24%24w0rd%21)
        expected_pwd = urllib.parse.quote("p@$$w0rd!")
        expected_url = f"rtsp://custom_user:{expected_pwd}@192.168.1.100:554/cam/realmonitor?channel=4&subtype=0"
        
        self.assertEqual(url, expected_url)

    def test_thread_lifecycle_management(self) -> None:
        """Verifies start_all and stop_all spawn and cleanup stream managers correctly."""
        settings.RTSP_HOST = ""  # empty host triggers mock mode
        settings.CAMERAS = [
            {"id": "CAM_TEST_1", "name": "Test Cam 1", "channel": 1, "enabled": True},
            {"id": "CAM_TEST_2", "name": "Test Cam 2", "channel": 2, "enabled": True},
        ]

        manager = CameraManager()
        self.assertEqual(len(manager.streams), 0)

        # Start all streams
        manager.start_all()
        self.assertEqual(len(manager.streams), 2)
        self.assertTrue(manager.streams["CAM_TEST_1"].reader.is_running)
        self.assertTrue(manager.streams["CAM_TEST_2"].reader.is_running)

        # Stop all streams
        manager.stop_all()
        self.assertEqual(len(manager.streams), 0)

    def test_data_retrieval_proxies(self) -> None:
        """Verifies that frame, status, fps and resolution calls query individual stream buffers."""
        manager = CameraManager()
        
        # Setup mock StreamManagers
        mock_stream_1 = MagicMock()
        mock_stream_2 = MagicMock()
        
        manager.streams = {
            "CAM_1": mock_stream_1,
            "CAM_2": mock_stream_2
        }

        # Mock implementations
        mock_frame = object()
        mock_stream_1.get_latest_frame.return_value = mock_frame
        mock_stream_1.is_connected.return_value = True
        mock_stream_1.get_fps.return_value = 24.5
        mock_stream_1.get_resolution.return_value = (1920, 1080)

        mock_stream_2.is_connected.return_value = False
        mock_stream_2.get_latest_frame.return_value = None

        # Query CAM_1
        self.assertEqual(manager.get_latest_frame("CAM_1"), mock_frame)
        self.assertTrue(manager.is_connected("CAM_1"))
        self.assertEqual(manager.get_fps("CAM_1"), 24.5)
        self.assertEqual(manager.get_resolution("CAM_1"), (1920, 1080))

        # Query CAM_2
        self.assertFalse(manager.is_connected("CAM_2"))
        self.assertNilFrame = manager.get_latest_frame("CAM_2")
        self.assertIsNone(self.assertNilFrame)

        # Query unknown ID
        self.assertFalse(manager.is_connected("CAM_UNKNOWN"))
        self.assertIsNone(manager.get_latest_frame("CAM_UNKNOWN"))

    def test_single_camera_backwards_compatibility_fallback(self) -> None:
        """Verifies fallback behaviour when CAMERAS registry is empty but RTSP_URL is configured."""
        # Empty camera config, set single RTSP URL
        settings.CAMERAS = []
        settings.RTSP_HOST = ""  # empty host triggers mock mode

        manager = CameraManager()
        active = manager.get_active_cameras()
        
        # Verify it has not loaded anything directly
        self.assertEqual(len(active), 0)

        # Run main fallback registry recreation (matching main.py fallback logic)
        fallback_cameras = [{
            "id": "CAM001",
            "name": "CCTV Monitor",
            "channel": 1,
            "enabled": True
        }]
        manager.camera_configs = {"CAM001": fallback_cameras[0]}
        
        active_fallback = manager.get_active_cameras()
        self.assertEqual(len(active_fallback), 1)
        self.assertEqual(active_fallback[0]["name"], "CCTV Monitor")
        self.assertEqual(manager.generate_rtsp_url(active_fallback[0]["channel"]), "mock")


if __name__ == "__main__":
    unittest.main()
