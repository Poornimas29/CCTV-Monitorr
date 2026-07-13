import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2

from ai.face_recognition import FaceRecognitionEngine


def write_test_image(path: Path, color: tuple[int, int, int]):
    image = np.full((120, 120, 3), color, dtype=np.uint8)
    image[30:90, 35:85] = (255, 255, 255)
    image[40:80, 42:78] = color
    cv2.imwrite(str(path), image)


def main():
    temp_dir = Path(tempfile.mkdtemp())
    employee_dir = temp_dir / "employee_images" / "EMP001"
    employee_dir.mkdir(parents=True, exist_ok=True)
    img_path = employee_dir / "img1.png"
    write_test_image(img_path, (20, 40, 60))

    manager = SimpleNamespace(
        get_all_employees=lambda: [
            {
                "employee_id": "EMP001",
                "name": "Rahul",
                "image_folder_abs": str(employee_dir),
            }
        ],
        get_employee_images=lambda employee_id: [str(img_path)],
    )

    engine = FaceRecognitionEngine(project_root=str(temp_dir), cache_path=str(temp_dir / "face_cache.json"), threshold=0.35, debug=True)
    summary = engine.initialize(manager)
    print("Initialization summary:", summary)

    frame = np.full((240, 320, 3), (20, 40, 60), dtype=np.uint8)
    frame[60:140, 90:190] = (255, 255, 255)
    frame[80:120, 110:170] = (20, 40, 60)

    print("Running recognize_frame...")
    result = engine.recognize_frame(frame)
    print("Result:", result)


if __name__ == '__main__':
    main()
