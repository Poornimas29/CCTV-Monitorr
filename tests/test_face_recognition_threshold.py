import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.face_recognition import FaceRecognitionEngine


class TestFaceRecognitionThreshold(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_root = Path(self.temp_dir.name)
        self.cache_path = self.project_root / "face_cache.json"

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
            get_employee_images=lambda employee_id: [str(employee_dir / "img1.png")],
        )

    def test_unknown_when_similarity_is_below_threshold(self) -> None:
        manager = self._make_employee_manager()
        engine = FaceRecognitionEngine(project_root=self.project_root, cache_path=self.cache_path, threshold=0.95, debug=False)
        engine.initialize(manager)

        frame = np.full((240, 320, 3), (120, 120, 120), dtype=np.uint8)
        result = engine.recognize_frame(frame)

        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["matched"])

    def test_no_face_detected_returns_no_face_status(self) -> None:
        manager = self._make_employee_manager()
        engine = FaceRecognitionEngine(project_root=self.project_root, cache_path=self.cache_path, threshold=0.95, debug=False)
        engine.initialize(manager)

        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        result = engine.recognize_frame(frame)

        self.assertEqual(result["status"], "no_face")


if __name__ == "__main__":
    unittest.main()
