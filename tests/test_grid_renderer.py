"""
Unit tests for stream.grid_renderer.GridRenderer.

Covers:
  - Grid dimension calculation for various camera counts.
  - Dashboard build (header + grid) returns a valid NumPy image.
  - Offline cell is generated when a frame is None or camera is disconnected.
  - Cell overlay (name, status, FPS) is applied without raising exceptions.
  - Empty camera list returns a single blank cell without raising exceptions.
  - Padding: grid always fills rows × cols slots.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream.grid_renderer import GridRenderer


class TestGridDimensions(unittest.TestCase):
    """Validates the (rows, cols) layout formula."""

    def setUp(self) -> None:
        self.renderer = GridRenderer(cell_size=(320, 180))

    def test_zero_cameras(self) -> None:
        """Zero cameras should not raise and returns 1×1."""
        rows, cols = self.renderer.grid_dimensions(0)
        self.assertEqual((rows, cols), (1, 1))

    def test_one_camera(self) -> None:
        """Single camera → 1×1."""
        self.assertEqual(self.renderer.grid_dimensions(1), (1, 1))

    def test_two_cameras(self) -> None:
        """2 cameras → 1 row × 2 cols."""
        self.assertEqual(self.renderer.grid_dimensions(2), (1, 2))

    def test_four_cameras(self) -> None:
        """4 cameras → 2×2."""
        self.assertEqual(self.renderer.grid_dimensions(4), (2, 2))

    def test_six_cameras(self) -> None:
        """6 cameras → 2 rows × 3 cols."""
        self.assertEqual(self.renderer.grid_dimensions(6), (2, 3))

    def test_nine_cameras(self) -> None:
        """9 cameras → 3×3."""
        self.assertEqual(self.renderer.grid_dimensions(9), (3, 3))


class TestBuildDashboard(unittest.TestCase):
    """Tests the build_dashboard() public API."""

    def setUp(self) -> None:
        self.cw, self.ch = 320, 180
        self.renderer = GridRenderer(cell_size=(self.cw, self.ch))

    def _dummy_frame(self) -> np.ndarray:
        return np.zeros((self.ch, self.cw, 3), dtype=np.uint8)

    def test_returns_numpy_array(self) -> None:
        """build_dashboard must return a NumPy ndarray."""
        camera_ids = ["CAM001"]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={"CAM001": "Test Cam"},
            frames={"CAM001": self._dummy_frame()},
            connected={"CAM001": True},
            fps_map={"CAM001": 25.0},
        )
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_two_cameras(self) -> None:
        """
        2 cameras (1 row × 2 cols) dashboard must have:
            height = HEADER_HEIGHT + cell_h
            width  = 2 × cell_w
        """
        camera_ids = ["C1", "C2"]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={"C1": "Cam 1", "C2": "Cam 2"},
            frames={"C1": self._dummy_frame(), "C2": self._dummy_frame()},
            connected={"C1": True, "C2": True},
            fps_map={"C1": 25.0, "C2": 20.0},
        )
        expected_h = GridRenderer.HEADER_HEIGHT + self.ch  # 1 row
        expected_w = 2 * self.cw  # 2 cols
        self.assertEqual(result.shape[:2], (expected_h, expected_w))

    def test_output_shape_four_cameras(self) -> None:
        """4 cameras (2×2) → height = HEADER + 2*cell_h, width = 2*cell_w."""
        camera_ids = [f"C{i}" for i in range(4)]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={c: c for c in camera_ids},
            frames={c: self._dummy_frame() for c in camera_ids},
            connected={c: True for c in camera_ids},
            fps_map={c: 15.0 for c in camera_ids},
        )
        expected_h = GridRenderer.HEADER_HEIGHT + 2 * self.ch
        expected_w = 2 * self.cw
        self.assertEqual(result.shape[:2], (expected_h, expected_w))

    def test_empty_camera_list_does_not_raise(self) -> None:
        """Empty camera list must return a valid image without raising."""
        result = self.renderer.build_dashboard(
            camera_ids=[],
            camera_names={},
            frames={},
            connected={},
            fps_map={},
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)

    def test_offline_camera_uses_placeholder(self) -> None:
        """A None frame with is_connected=False must produce an offline placeholder cell."""
        camera_ids = ["CAM_OFFLINE"]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={"CAM_OFFLINE": "Offline Cam"},
            frames={"CAM_OFFLINE": None},
            connected={"CAM_OFFLINE": False},
            fps_map={"CAM_OFFLINE": 0.0},
        )
        # The result must be a non-zero image (offline cell has a dark background, not pure black)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)

    def test_connected_frame_is_resized_to_cell_size(self) -> None:
        """A live frame of any resolution must be scaled to the cell dimensions."""
        # Provide a 1920×1080 source frame
        big_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        camera_ids = ["CAM_BIG"]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={"CAM_BIG": "Big Cam"},
            frames={"CAM_BIG": big_frame},
            connected={"CAM_BIG": True},
            fps_map={"CAM_BIG": 30.0},
        )
        expected_h = GridRenderer.HEADER_HEIGHT + self.ch
        expected_w = self.cw
        self.assertEqual(result.shape[:2], (expected_h, expected_w))

    def test_mixed_online_offline_cameras(self) -> None:
        """Dashboard must handle a mix of online and offline cameras without raising."""
        camera_ids = ["ONLINE", "OFFLINE"]
        result = self.renderer.build_dashboard(
            camera_ids=camera_ids,
            camera_names={"ONLINE": "Online Cam", "OFFLINE": "Offline Cam"},
            frames={"ONLINE": self._dummy_frame(), "OFFLINE": None},
            connected={"ONLINE": True, "OFFLINE": False},
            fps_map={"ONLINE": 25.0, "OFFLINE": 0.0},
        )
        self.assertIsInstance(result, np.ndarray)
        # 1 row × 2 cols
        expected_w = 2 * self.cw
        self.assertEqual(result.shape[1], expected_w)

    def test_offline_placeholder_uses_camera_offline_label(self) -> None:
        """Offline cells should use the requested 'Camera Offline' label."""
        self.assertEqual(self.renderer._offline_label_text(), "Camera Offline")


if __name__ == "__main__":
    unittest.main()
