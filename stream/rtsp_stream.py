# stream/rtsp_stream.py
"""RTSP video stream wrapper.

Provides a lightweight class that opens an RTSP URL with OpenCV, attempts
reconnection on failure, and yields frames together with timestamps.
Supports a premium synthetic mock mode when the URL is set to "mock".
"""

import cv2
import time
import numpy as np
from datetime import datetime
from typing import Tuple

class RTSPStream:
    """Handle a single RTSP camera.

    Parameters
    ----------
    cam_id: str
        Identifier for the camera (used for logging and output).
    url: str
        RTSP URL.
    reconnect_delay: int, optional
        Seconds to wait before trying to reconnect after a failure.
    """

    def __init__(self, cam_id: str, url: str, reconnect_delay: int = 5):
        self.cam_id = cam_id
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap: cv2.VideoCapture | None = None
        self._stop = False
        self._frame_interval: float = 0.0   # seconds between frames for file playback
        self._last_frame_time: float = 0.0  # monotonic clock of last decoded frame

        # Enable mock simulation mode if the URL is "mock"
        self.is_mock = (url.lower() == "mock")
        # Detect local file playback (not RTSP and not mock)
        self._is_file = (
            not self.is_mock
            and not url.lower().startswith("rtsp://")
        )

    def _open(self) -> None:
        if self.is_mock:
            return

        # Open RTSP using FFMPEG backend.
        # Set stimeout (socket timeout) to 5 seconds so a dead RTSP host fails
        # quickly instead of blocking the reader thread for 20–30 seconds.
        # The option is embedded in the RTSP URL as an FFMPEG AVOption.
        timeout_url = self.url
        # Only modify URL if it is an RTSP stream.
        if timeout_url.lower().startswith('rtsp://'):
            if "?" in timeout_url:
                timeout_url = self.url + "&timeout=5000000"  # 5s in microseconds
            else:
                timeout_url = self.url + "?timeout=5000000"
        # For local file paths (e.g., playback video), use the original URL.
        self.cap = cv2.VideoCapture(timeout_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            # Fallback: try without the custom timeout URL
            self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise RuntimeError(f"[{self.cam_id}] Unable to open RTSP stream: {self.url}")
        # Set video buffer size to 1 to enforce real-time decoding and prevent queue buildup lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # For local file playback, read the native FPS and compute per-frame sleep interval.
        # This prevents the reader thread from decoding frames faster than real-time.
        if self._is_file:
            native_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if native_fps and native_fps > 0:
                self._frame_interval = 1.0 / native_fps
            else:
                self._frame_interval = 1.0 / 25.0   # fallback: 25 fps
            self._last_frame_time = 0.0


    def read(self) -> Tuple[bool, "any", datetime]:
        """Read a single frame.

        Returns
        -------
        tuple
            ``(success, frame, timestamp)`` where ``timestamp`` is a ``datetime``
            in UTC.
        """
        if self.is_mock:
            return self._read_mock()

        if self.cap is None or not self.cap.isOpened():
            self._open()

        # Throttle file playback to native video FPS so the dashboard
        # plays at real-time speed instead of racing through frames.
        if self._is_file and self._frame_interval > 0:
            now = time.monotonic()
            elapsed = now - self._last_frame_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)
            self._last_frame_time = time.monotonic()

        ret, frame = self.cap.read()
        if not ret:
            # End of file — loop back to the start for continuous playback
            if self._is_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                # Force reconnection on next call.
                self.cap.release()
                self.cap = None
                raise RuntimeError(f"[{self.cam_id}] Frame read failed")
        return True, frame, datetime.utcnow()

    def _read_mock(self) -> Tuple[bool, np.ndarray, datetime]:
        """Generate a simulated camera frame for mock testing."""
        w, h = 640, 360
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. Premium dark-green grid background
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (12, 22, 12), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (12, 22, 12), 1)
            
        # 2. Radar sweep graphics
        cx, cy = w // 2, h // 2
        r = 150
        cv2.circle(frame, (cx, cy), r, (0, 60, 0), 2)
        cv2.circle(frame, (cx, cy), r // 2, (0, 40, 0), 1)
        
        # Calculate sweeping line angle
        t = time.time()
        angle = (t * 1.5) % (2 * np.pi)
        
        end_x = int(cx + r * np.cos(angle))
        end_y = int(cy + r * np.sin(angle))
        cv2.line(frame, (cx, cy), (end_x, end_y), (0, 180, 0), 2)
        
        # 3. Draw walking targets corresponding to YOLO26Detector simulation bboxes
        tx1 = int((w * 0.3) + (w * 0.1) * np.sin(t * 0.4))
        ty1 = int((h * 0.55) + 20 * np.cos(t * 0.4))
        
        # Draw target indicators
        cv2.circle(frame, (tx1, ty1), 6, (0, 255, 0), -1)
        cv2.circle(frame, (tx1, ty1), 12, (0, 255, 0), 1)
        
        tx2 = int(w * 0.74)
        ty2 = int(h * 0.55)
        cv2.circle(frame, (tx2, ty2), 6, (0, 255, 0), -1)
        
        # 4. Premium HUD Label
        cv2.putText(frame, f"LIVE MOCK - {self.cam_id}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Simulate frame timing (25 FPS -> 40ms)
        time.sleep(0.04)
        
        return True, frame, datetime.utcnow()

    def release(self) -> None:
        self._stop = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
