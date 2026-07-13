# stream/frame_buffer.py
"""Thread‑safe frame buffer for a single camera.

The buffer stores the most recent frames up to a configurable maximum.
It is used by the ``CameraManager`` to allow async consumption of frames
without dropping the newest image.
"""

from collections import deque
from threading import Lock
from typing import Deque, Tuple
import numpy as np
from datetime import datetime

class FrameBuffer:
    """Simple ring buffer for frames.

    Parameters
    ----------
    maxlen: int, optional
        Maximum number of frames to keep. ``0`` means unlimited.
    """
    def __init__(self, maxlen: int = 10):
        self.buffer: Deque[Tuple[np.ndarray, datetime]] = deque(maxlen=maxlen)
        self.lock = Lock()

    def add(self, frame: np.ndarray, timestamp: datetime) -> None:
        with self.lock:
            self.buffer.append((frame, timestamp))

    def latest(self) -> Tuple[np.ndarray, datetime] | None:
        """Return the most recent frame, or ``None`` if empty."""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def put(self, frame: np.ndarray) -> None:
        """Add a copy of the frame to the buffer (compatibility with legacy code)."""
        self.add(frame.copy(), datetime.utcnow())

    def get(self) -> np.ndarray | None:
        """Return the most recent frame, or None if empty (compatibility with legacy code)."""
        latest = self.latest()
        if latest is None:
            return None
        frame, _ = latest
        return frame

    def clear(self) -> None:
        """Clear all frames from the buffer (compatibility with legacy code)."""
        with self.lock:
            self.buffer.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)
