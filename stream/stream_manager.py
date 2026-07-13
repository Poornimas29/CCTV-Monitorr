# stream/stream_manager.py
"""StreamManager manages a single RTSP stream.

It creates a background thread that reads frames from the :class:`RTSPStream`
wrapper and stores the most recent frame in a :class:`FrameBuffer`.  The
public API mirrors the legacy ``stream_manager`` used elsewhere in the project:

* ``start()`` – launch the reader thread
* ``stop()`` – stop the thread and release the OpenCV capture
* ``get_latest_frame()`` – obtain the newest frame (or ``None``)
* ``is_connected()`` – whether the stream is currently delivering frames
* ``get_fps()`` – measured input FPS
* ``get_resolution()`` – width, height of the latest frame
"""

import logging
import time
from datetime import datetime
from threading import Event, Thread
from typing import Optional, Tuple

import numpy as np

from .frame_buffer import FrameBuffer
from .rtsp_stream import RTSPStream

logger = logging.getLogger(__name__)


class StreamManager:
    """Manage a single RTSP stream and expose a simple, thread‑safe API.

    The class is deliberately lightweight – it does **not** perform any image
    processing.  All heavy work (detection, tracking, etc.) is done elsewhere.
    """

    def __init__(self, rtsp_url: str, reconnect_interval: int = 5, camera_name: Optional[str] = None) -> None:
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval
        self.camera_name = camera_name or "Camera"
        # Frame buffer stores only the most recent frame.
        self.frame_buffer = FrameBuffer(maxlen=1)
        # RTSPStream expects a cam_id and url.
        self._stream = RTSPStream(self.camera_name, self.rtsp_url, reconnect_delay=self.reconnect_interval)
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._connected = False
        self._last_fps: float = 0.0
        self._last_resolution: Tuple[int, int] = (0, 0)

    # ---------------------------------------------------------------------
    # Lifecycle control
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Start the background reader thread.
        Re‑starts automatically if called after ``stop()``.
        """
        if self._thread and self._thread.is_alive():
            logger.warning("StreamManager thread already running for %s", self.camera_name)
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True, name=f"{self.camera_name}-reader")
        self._thread.start()
        logger.info("StreamManager started for %s", self.camera_name)

    def stop(self) -> None:
        """Signal the thread to exit and release the underlying capture object."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._stream.release()
        self._connected = False
        logger.info("StreamManager stopped for %s", self.camera_name)

    # ---------------------------------------------------------------------
    # Internal reader loop
    # ---------------------------------------------------------------------
    def _run(self) -> None:
        """Continuously read frames, update buffer, and compute FPS.
        Errors are logged and cause a short sleep before retrying.
        """
        frame_counter = 0
        fps_start = None
        while not self._stop_event.is_set():
            try:
                success, frame, _ = self._stream.read()
                if success:
                    self._connected = True
                    # Store latest frame with timestamp.
                    self.frame_buffer.add(frame, datetime.utcnow())
                    h, w = frame.shape[:2]
                    self._last_resolution = (w, h)
                    # Simple FPS calculation.
                    if fps_start is None:
                        fps_start = datetime.utcnow()
                    frame_counter += 1
                    elapsed = (datetime.utcnow() - fps_start).total_seconds()
                    if elapsed >= 1.0:
                        self._last_fps = frame_counter / elapsed
                        frame_counter = 0
                        fps_start = datetime.utcnow()
                else:
                    self._connected = False
            except Exception as exc:
                logger.error("[%s] Stream read error: %s", self.camera_name, exc)
                self._connected = False
                time.sleep(self.reconnect_interval)
        # Clean exit.
        self._connected = False

    # ---------------------------------------------------------------------
    # Public query methods
    # ---------------------------------------------------------------------
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame or ``None`` if there is none yet."""
        latest = self.frame_buffer.latest()
        if latest is None:
            return None
        frame, _ = latest
        return frame

    def is_connected(self) -> bool:
        """True when the stream is currently delivering frames."""
        return self._connected

    def get_fps(self) -> float:
        """Current measured input FPS (0.0 if not enough data)."""
        return self._last_fps

    def get_resolution(self) -> Tuple[int, int]:
        """Resolution of the most recent frame as ``(width, height)``."""
        return self._last_resolution
