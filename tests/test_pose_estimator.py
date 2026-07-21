# tests/test_pose_estimator.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Define mock mediapipe structures before importing the estimator
mock_mp = MagicMock()
mock_python = MagicMock()
mock_vision = MagicMock()

mock_mp.tasks = mock_python
mock_python.python = mock_python
mock_python.vision = mock_vision
mock_python.BaseOptions = MagicMock()
mock_vision.PoseLandmarkerOptions = MagicMock()
mock_vision.RunningMode.IMAGE = 'IMAGE'

sys.modules['mediapipe'] = mock_mp
sys.modules['mediapipe.tasks'] = mock_python
sys.modules['mediapipe.tasks.python'] = mock_python
sys.modules['mediapipe.tasks.python.vision'] = mock_vision

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.pose_estimator import MediaPipePoseEstimator

class TestPoseEstimator(unittest.TestCase):
    
    def setUp(self) -> None:
        self.exists_patcher = patch('ai.pose_estimator.os.path.exists', return_value=True)
        self.exists_patcher.start()
        mock_vision.PoseLandmarker.create_from_options.reset_mock()
        # Reset the detect mock
        mock_landmarker = mock_vision.PoseLandmarker.create_from_options.return_value
        mock_landmarker.detect = MagicMock()

    def tearDown(self) -> None:
        self.exists_patcher.stop()

    def test_pose_estimator_init(self) -> None:
        estimator = MediaPipePoseEstimator()
        self.assertIsNotNone(estimator.landmarker)
        mock_vision.PoseLandmarker.create_from_options.assert_called_once()

    def test_estimate_pose_no_landmarks(self) -> None:
        mock_landmarker = mock_vision.PoseLandmarker.create_from_options.return_value
        mock_landmarker.detect.return_value.pose_landmarks = None
        
        estimator = MediaPipePoseEstimator()
        estimator.use_fallback = False
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = [10, 10, 90, 90]
        
        res = estimator.estimate_pose(frame, bbox)
        self.assertIsNone(res)

    def test_estimate_pose_with_landmarks(self) -> None:
        # Create mock landmarks list of size 33
        mock_landmarks = []
        for idx in range(33):
            lm = MagicMock()
            if idx == 0:  # Nose
                lm.x = 0.5
                lm.y = 0.3
            elif idx == 11:  # Left shoulder
                lm.x = 0.3
                lm.y = 0.5
            elif idx == 12:  # Right shoulder
                lm.x = 0.7
                lm.y = 0.5
            elif idx == 15:  # Left hand wrist
                lm.x = 0.2
                lm.y = 0.7
            elif idx == 16:  # Right hand wrist
                lm.x = 0.8
                lm.y = 0.7
            elif idx == 23:  # Left hip
                lm.x = 0.3
                lm.y = 0.8
            elif idx == 24:  # Right hip
                lm.x = 0.7
                lm.y = 0.8
            else:
                lm.x = 0.5
                lm.y = 0.5
            lm.visibility = 0.9
            mock_landmarks.append(lm)

        mock_landmarker = mock_vision.PoseLandmarker.create_from_options.return_value
        mock_landmarker.detect.return_value.pose_landmarks = [mock_landmarks]
        
        estimator = MediaPipePoseEstimator()
        estimator.use_fallback = False
        import numpy as np
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = [50, 50, 150, 150]
        
        res = estimator.estimate_pose(frame, bbox)
        
        self.assertIsNotNone(res)
        self.assertEqual(res["head_direction"], "Front")
        self.assertTrue(res["is_stable"])
        
        # Verify global coordinate translations
        # nose crop x is 0.5, translated: 40 + 0.5 * 120 = 100
        self.assertEqual(res["landmarks"][0]["x"], 100)
        # nose crop y is 0.3, translated: 40 + 0.3 * 120 = 76
        self.assertEqual(res["landmarks"][0]["y"], 76)
        
        # Verify shoulder points
        self.assertIsNotNone(res["shoulders"]["left"])
        self.assertIsNotNone(res["shoulders"]["right"])
        
        # Verify hand points
        self.assertIsNotNone(res["hands"]["left"])
        self.assertIsNotNone(res["hands"]["right"])

if __name__ == "__main__":
    unittest.main()
