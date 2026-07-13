# stream/camera_manager.py
"""CameraManager orchestrates multiple RTSP streams.

It now supports two usage patterns for backward compatibility:

1. **Legacy** – ``CameraManager(reconnect_interval=RECONNECT_INTERVAL)`` as used in
   the original ``main.py``.  In this case the manager reads the camera list from
   ``config.settings.CAMERAS``.
2. **Explicit** – ``CameraManager(camera_cfg, buffer_size=10)`` where a mapping
   ``camera_id -> rtsp_url`` is supplied directly.

Both patterns expose the same public API: ``start()``, ``stop()``,
``read_frames()`` and ``get_active_cameras()``.
"""

import threading
import time
from queue import Queue, Empty
from datetime import datetime
from typing import Dict, Tuple, List

from .rtsp_stream import RTSPStream
from .frame_buffer import FrameBuffer

# Load default configuration when the caller does not provide one.
try:
    from config.settings import CAMERAS as DEFAULT_CAMERAS  # type: ignore
except Exception:
    DEFAULT_CAMERAS = []  # Fallback if settings cannot be imported.


class CameraManager:
    """Manage a collection of RTSP cameras.

    Parameters
    ----------
    camera_cfg: Dict[str, str] | None
        Mapping ``camera_id -> rtsp_url``. If ``None`` the manager falls back to
        ``config.settings.CAMERAS`` (a list of dicts with ``id`` and ``url``).
    reconnect_interval: int, optional
        Seconds to wait before trying to reconnect after a failure (legacy
        compatibility argument).
    buffer_size: int, optional
        Maximum number of frames stored per camera (default 10).
    """

    def __init__(
        self,
        camera_cfg: Dict[str, str] | None = None,
        reconnect_interval: int = 5,
        buffer_size: int = 10,
    ):
        # Resolve configuration.
        if camera_cfg is None:
            # If the static CAMERAS list defines enabled cameras, use them.
            if any(cam.get("enabled", False) for cam in DEFAULT_CAMERAS):
                camera_cfg = {
                    cam["id"]: f"rtsp://{cam.get('username', '')}:{cam.get('password', '')}@{cam.get('host', 'localhost')}:{cam.get('port', 554)}/{cam.get('path', '')}"  # noqa: E501
                    for cam in DEFAULT_CAMERAS
                    if cam.get("enabled", False)
                }
            else:
                # No static cameras – build three identical streams from env vars.
                from config.settings import build_default_rtsp_url
                default_url = build_default_rtsp_url()
                camera_cfg = {f"CAM{i+1:03d}": default_url for i in range(3)}
        self.camera_cfg = camera_cfg
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size
        self._queue: Queue = Queue(maxsize=50)
        # Store StreamManager instances per camera.
        self._streams: Dict[str, "StreamManager"] = {}
        # Frame buffers are now managed by each StreamManager; keep placeholder dict for API compatibility.
        self._buffers: Dict[str, None] = {}
        self._threads: List[threading.Thread] = []  # retained for potential future use
        self._stop_event = threading.Event()
    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def start_all(self) -> None:
        """Start all RTSP streams and the consumer thread.

        Uses :class:`StreamManager` to handle each individual stream, providing
        connectivity status, FPS measurement, and resolution queries.
        """
        from .stream_manager import StreamManager  # local import to avoid circular
        for cam_id, url in self.camera_cfg.items():
            # Initialise a StreamManager for each camera.
            stream_mgr = StreamManager(rtsp_url=url, reconnect_interval=self.reconnect_interval, camera_name=cam_id)
            stream_mgr.start()
            self._streams[cam_id] = stream_mgr
            # No separate FrameBuffer needed; StreamManager already buffers the latest frame.
            self._buffers[cam_id] = None  # placeholder to keep API compatibility
            # Threads are managed by StreamManager, so we don't spawn producer threads here.


    def start(self) -> None:
        """Start all RTSP streams and the consumer thread. (alias for start_all)"""
        self.start_all()

    def stop_all(self) -> None:
        """Signal all threads to stop and release resources."""
        self._stop_event.set()
        for s in self._streams.values():
            s.release()
        # Daemon threads will exit automatically.

    # ---------------------------------------------------------------------
    # Internal producer per camera
    # ---------------------------------------------------------------------
    def _producer(self, cam_id: str, stream: RTSPStream) -> None:
        """Continuously read frames from *stream* and put them on the shared queue."""
        while not self._stop_event.is_set():
            try:
                success, frame, ts = stream.read()
                if not success:
                    continue
                self._buffers[cam_id].add(frame, ts)
                self._queue.put((cam_id, frame, ts))
            except Exception as exc:
                print(f"[CameraManager] {cam_id} error: {exc}")
                time.sleep(self.reconnect_interval)

    # ---------------------------------------------------------------------
    # Consumer API
    # ---------------------------------------------------------------------
    def read_frames(self):
        """Generator yielding ``(camera_id, frame, timestamp)`` tuples.

        It blocks only for a short timeout; if no frame is available it simply
        retries, keeping the caller responsive.
        """
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                yield item
            except Empty:
                continue

    def get_resolution(self, cam_id: str) -> tuple[int, int]:
        """Return the resolution (width, height) of the latest frame for *cam_id*.
        Delegates to the underlying :class:`StreamManager`.
        """
        stream_mgr = self._streams.get(cam_id)
        if not stream_mgr:
            return (0, 0)
        return stream_mgr.get_resolution()

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def get_active_cameras(self) -> List[Dict[str, str]]:
        """Return a list of camera dictionaries compatible with the legacy code.

        The original ``main.py`` expects a list of dicts with keys ``id`` and
        ``name`` (and optionally ``channel``).  We synthesize the ``name`` from
        the configuration if available; otherwise we fall back to the ``id``.
        """
        cameras = []
        for cam_id in self.camera_cfg.keys():
            cameras.append({"id": cam_id, "name": cam_id})
        return cameras

    # ---------------------------------------------------------------------
    # New API delegations used by ``main.py``
    # ---------------------------------------------------------------------
    def get_latest_frame(self, cam_id: str) -> "np.ndarray | None":
        """Return the most recent frame for *cam_id* or ``None`` if unavailable."""
        stream_mgr = self._streams.get(cam_id)
        if not stream_mgr:
            return None
        return stream_mgr.get_latest_frame()

    def is_connected(self, cam_id: str) -> bool:
        """Return ``True`` if *cam_id* is currently delivering frames."""
        stream_mgr = self._streams.get(cam_id)
        if not stream_mgr:
            return False
        return stream_mgr.is_connected()

    def get_fps(self, cam_id: str) -> float:
        """Return the measured input FPS for *cam_id* (0.0 if not enough data)."""
        stream_mgr = self._streams.get(cam_id)
        if not stream_mgr:
            return 0.0
        return stream_mgr.get_fps()
