import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.face_recognition import FaceRecognitionEngine


class TestFaceRecognitionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_root = Path(self.temp_dir.name)
        self.cache_path = self.project_root / "face_cache.json"

        # Mock _detect_faces to simulate InsightFace 512-d output
        def mock_detect_faces(engine_instance, frame):
            if frame is None or frame.size == 0 or frame.mean() < 5.0:
                return []
            corner_color = frame[0, 0]
            if isinstance(corner_color, np.ndarray):
                colors = [int(v) for v in corner_color]
            else:
                colors = [int(corner_color)] * 3
            # Sort and divide by 10 for channel-swapping and compression robustness
            r, g, b = sorted([int(round(c / 10.0)) for c in colors])
            seed = (r * 10000 + g * 100 + b) % 65535
            state = np.random.RandomState(seed)
            mock_embedding = state.randn(512).astype(np.float32)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            return [{
                "bbox": (0, 0, frame.shape[1], frame.shape[0]),
                "face": frame,
                "embedding": mock_embedding,
                "aimg": frame
            }]

        self.original_detect = FaceRecognitionEngine._detect_faces
        FaceRecognitionEngine._detect_faces = mock_detect_faces

    def tearDown(self) -> None:
        FaceRecognitionEngine._detect_faces = self.original_detect

    def _write_test_image(self, path: Path, color: tuple[int, int, int]) -> None:
        image = np.full((120, 120, 3), color, dtype=np.uint8)
        image[30:90, 35:85] = (255, 255, 255)
        image[40:80, 42:78] = color
        cv2 = __import__("cv2")
        cv2.imwrite(str(path), image)

    def _make_employee_manager(self) -> SimpleNamespace:
        employee_dir = self.project_root / "employee_images" / "EMP001"
        employee_dir.mkdir(parents=True, exist_ok=True)
        self._write_test_image(employee_dir / "img1.png", (20, 40, 60))

        return SimpleNamespace(
            get_all_employees=lambda: [
                {
                    "employee_id": "EMP001",
                    "name": "Rahul",
                    "image_folder_abs": str(employee_dir),
                }
            ],
            get_employee_images=lambda employee_id: [
                str(employee_dir / "img1.png")
            ],
        )

    def test_initialize_loads_embeddings_and_reports_summary(self) -> None:
        manager = self._make_employee_manager()
        engine = FaceRecognitionEngine(project_root=self.project_root, cache_path=self.cache_path)

        summary = engine.initialize(manager)

        self.assertEqual(summary["registered_employees"], 1)
        self.assertEqual(summary["total_face_images"], 1)
        self.assertEqual(summary["embeddings_loaded"], True)
        self.assertTrue(self.cache_path.exists())

    def test_process_frame_recognizes_matching_employee(self) -> None:
        manager = self._make_employee_manager()
        engine = FaceRecognitionEngine(project_root=self.project_root, cache_path=self.cache_path, threshold=0.35, debug=False)
        engine.initialize(manager)

        frame = np.full((240, 320, 3), (20, 40, 60), dtype=np.uint8)
        frame[60:140, 90:190] = (255, 255, 255)
        frame[80:120, 110:170] = (20, 40, 60)

        result = engine.recognize_frame(frame)

        self.assertEqual(result["employee_id"], "EMP001")
        self.assertEqual(result["employee_name"], "Rahul")
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_debug_output_reports_decision_details(self) -> None:
        manager = self._make_employee_manager()
        engine = FaceRecognitionEngine(project_root=self.project_root, cache_path=self.cache_path, threshold=0.35, debug=True)
        engine.initialize(manager)

        frame = np.full((240, 320, 3), (20, 40, 60), dtype=np.uint8)
        frame[60:140, 90:190] = (255, 255, 255)
        frame[80:120, 110:170] = (20, 40, 60)

        with contextlib.redirect_stdout(io.StringIO()) as output:
            engine.recognize_frame(frame)

        text = output.getvalue()
        self.assertIn("Frame Received", text)
        self.assertIn("Number of Faces Detected", text)
        self.assertIn("Best Match", text)
        self.assertIn("Recognition Threshold", text)
        self.assertIn("Final Decision", text)


if __name__ == "__main__":
    unittest.main()
