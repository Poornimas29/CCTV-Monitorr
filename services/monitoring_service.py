# services/monitoring_service.py
"""MonitoringService orchestrates the end‑to‑end pipeline.

It ties together:
* :class:`stream.camera_manager.CameraManager` – reads frames from one or more RTSP sources.
* :class:`detection.yolo26_detector.YOLO26Detector` – runs a lightweight person detector.
* :class:`tracking.tracker.Tracker` – maintains persistent tracks via ByteTrack.
* :class:`person_management.person_manager.PersonManager` – produces per‑person state changes.
* :class:`visualisation.renderer.Renderer` – draws an annotated overlay on the frame.

The service writes a JSON payload for every processed frame into ``settings.OUTPUT_DIR``.
No external API is exposed at this stage – the service is started via a simple
script (``run_monitoring.py``)."""

import os
import json
import cv2
import argparse
import sys
# Ensure the project root (which contains the 'config' package) is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime
from typing import Optional

from config.settings import settings
from stream.camera_manager import CameraManager
from detection.yolo26_detector import YOLO26Detector
from tracking.tracker import Tracker
from person_management.person_manager import PersonManager
from visualisation.renderer import Renderer


class MonitoringService:
    def __init__(self, max_frames: Optional[int] = None, display: bool = False):
        self.max_frames = max_frames
        self.display = display
        self.frame_counter = 0

        # Initialise components.
        self.camera_manager = CameraManager()
        self.detector = YOLO26Detector.instance()
        self.tracker = Tracker()
        self.person_manager = PersonManager()
        self.renderer = Renderer()

        # Ensure output directory exists.
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    def _write_json(self, cam_id: str, ts: datetime, persons: list[dict]):
        payload = {
            "camera_id": cam_id,
            "timestamp": ts.isoformat(),
            "persons": persons,
        }
        filename = f"{cam_id}_{ts.strftime('%Y%m%d_%H%M%S_%f')}.json"
        out_path = os.path.join(settings.OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(self):
        """Start the processing loop.

        The loop runs until ``max_frames`` is reached (if set) or the user
        interrupts with ``Ctrl‑C``.  Frames are read from ``CameraManager`` as a
        generator yielding ``(cam_id, frame, ts)`` tuples.
        """
        self.camera_manager.start_all()
        try:
            for cam_id, frame, ts in self.camera_manager.read_frames():
                # Detection – returns a list of ``Detection`` objects.
                detections = self.detector.detect(frame)
                # Tracking – produces a list of ``Track`` objects.
                tracks = self.tracker.update(detections)
                # Person state – convert tracks to per‑person dicts.
                person_states = self.person_manager.process_tracks(tracks, ts)
                # Render annotation.
                annotated = self.renderer.draw(
                    frame=frame,
                    persons=person_states,
                    fps=self.tracker.get_fps(),
                )
                # Write JSON output.
                self._write_json(cam_id, ts, person_states)

                if self.display:
                    cv2.imshow(f"{cam_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.frame_counter += 1
                if self.max_frames is not None and self.frame_counter >= self.max_frames:
                    break
        finally:
            self.camera_manager.stop_all()
            if self.display:
                cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the CCTV monitoring pipeline.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process before exiting.")
    parser.add_argument("--display", action="store_true",
                        help="Show a live OpenCV window with annotations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    service = MonitoringService(max_frames=args.max_frames, display=args.display)
    service.run()
