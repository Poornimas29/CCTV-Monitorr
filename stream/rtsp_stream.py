# stream/rtsp_stream.py
"""RTSP video stream wrapper.

Provides a lightweight class that opens an RTSP URL with OpenCV, attempts
reconnection on failure, and yields frames together with timestamps.
"""

import cv2
import time
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

    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.cam_id}] Unable to open RTSP stream: {self.url}")

    def read(self) -> Tuple[bool, "any", datetime]:
        """Read a single frame.

        Returns
        -------
        tuple
            ``(success, frame, timestamp)`` where ``timestamp`` is a ``datetime``
            in UTC.
        """
        if self.cap is None or not self.cap.isOpened():
            self._open()
        ret, frame = self.cap.read()
        if not ret:
            # Force reconnection on next call.
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"[{self.cam_id}] Frame read failed")
        return True, frame, datetime.utcnow()

    def release(self) -> None:
        self._stop = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
