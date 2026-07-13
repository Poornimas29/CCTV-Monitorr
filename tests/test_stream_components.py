import os
import sys
import time
import unittest
import numpy as np

# Ensure project root is in sys.path for running tests in isolated environments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from stream.frame_buffer import FrameBuffer
from stream.reconnect_handler import ReconnectHandler


class TestStreamComponents(unittest.TestCase):
    """Test suite covering configurations and stream module components."""

    def test_settings_load(self) -> None:
        """Verifies settings load environment variables or use fallback defaults."""
        self.assertIsNotNone(settings.RTSP_URL)
        self.assertGreater(settings.RECONNECT_INTERVAL, 0)
        self.assertGreater(settings.TARGET_FPS, 0)
        self.assertIsNotNone(settings.CAPTURE_DIR)
        self.assertIsNotNone(settings.LOG_DIR)

    def test_frame_buffer_put_get(self) -> None:
        """Validates thread-safe buffer put, overwrite, clear, and get operations."""
        buffer = FrameBuffer()
        self.assertIsNone(buffer.get(), "Buffer must initially return None.")

        # Test putting simple frame
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame1[0, 0] = [10, 20, 30]
        buffer.put(frame1)

        result1 = buffer.get()
        self.assertIsNotNone(result1)
        # Ensure deep copy was made
        self.assertIsNot(result1, frame1)
        self.assertEqual(result1[0, 0].tolist(), [10, 20, 30])

        # Test overwrite / drop-old frame behaviour
        frame2 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2[0, 0] = [100, 150, 200]
        buffer.put(frame2)

        result2 = buffer.get()
        self.assertEqual(result2[0, 0].tolist(), [100, 150, 200])

        # Test clear
        buffer.clear()
        self.assertIsNone(buffer.get(), "Buffer must be None after clear.")

    def test_reconnect_handler(self) -> None:
        """Verifies retry counter increments, waits, and resets correctly."""
        handler = ReconnectHandler(reconnect_interval=1)
        self.assertEqual(handler.attempts, 0)
        self.assertFalse(handler.is_reconnecting)

        # Trigger reconnection status
        handler.start_reconnect()
        self.assertTrue(handler.is_reconnecting)

        # Measure sleeping interval
        start_time = time.time()
        handler.wait_and_retry()
        end_time = time.time()

        self.assertEqual(handler.attempts, 1)
        self.assertGreaterEqual(
            end_time - start_time, 0.95,
            "Should wait for configured reconnect interval duration."
        )

        # Reset connection metrics
        handler.reset()
        self.assertEqual(handler.attempts, 0)
        self.assertFalse(handler.is_reconnecting)


if __name__ == "__main__":
    unittest.main()
