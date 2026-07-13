# stream/camera_manager.py
"""CameraManager orchestrates multiple RTSP streams.

It supports two usage patterns for backward compatibility:

1. **Legacy** – ``CameraManager(reconnect_interval=RECONNECT_INTERVAL)`` as used in
   the original ``main.py``.  In this case the manager reads the camera list from
   ``config.settings.CAMERAS``.
2. **Explicit** – ``CameraManager(camera_cfg, buffer_size=10)`` where a mapping
   ``camera_id -> rtsp_url`` is supplied directly.

Both patterns expose the same public API: ``start()``, ``stop()``,
``read_frames()`` and ``get_active_cameras()``.
"""

import logging
import threading
import time
from queue import Queue, Empty
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import numpy as np

from .rtsp_stream import RTSPStream
from .frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)


class CameraManager:
    """Manage a collection of RTSP cameras.

    Parameters
    ----------
    camera_cfg: Dict[str, str] | None
        Mapping ``camera_id -> rtsp_url``. If ``None`` the manager falls back to
        ``config.settings.CAMERAS``.
    reconnect_interval: int, optional
        Seconds to wait before trying to reconnect after a failure.
    buffer_size: int, optional
        Maximum number of frames stored per camera.
    """

    def __init__(
        self,
        camera_cfg: Optional[Dict[str, str]] = None,
        reconnect_interval: int = 5,
        buffer_size: int = 10,
    ):
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size
        self._queue: Queue = Queue(maxsize=50)
        self.streams: Dict[str, "StreamManager"] = {}
        self._streams = self.streams
        self._buffers: Dict[str, None] = {}
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # Load configurations dynamically into self.camera_configs to preserve backward compatibility.
        if camera_cfg is not None:
            self.camera_configs = {
                cam_id: {
                    "id": cam_id,
                    "name": cam_id,
                    "channel": 1,
                    "enabled": True,
                    "url": url
                }
                for cam_id, url in camera_cfg.items()
            }
        else:
            from config import settings
            self.camera_configs = {
                cam["id"]: cam
                for cam in getattr(settings, "CAMERAS", [])
                if cam.get("enabled", False)
            }

        # Keep camera_cfg synced for any legacy direct access
        self.camera_cfg = {
            cam_id: cam.get("url") or self.generate_rtsp_url(cam.get("channel", 1))
            for cam_id, cam in self.camera_configs.items()
        }

    def generate_rtsp_url(self, channel: int) -> str:
        """Generate RTSP URL for a given channel."""
        import urllib.parse
        from config import settings
        
        # If RTSP_HOST is empty, or "mock", or RTSP_URL is "mock", return "mock"
        rtsp_host = getattr(settings, "RTSP_HOST", "")
        rtsp_url = getattr(settings, "RTSP_URL", "mock")
        if not rtsp_host or rtsp_host.lower() == "mock" or rtsp_url.lower() == "mock":
            return "mock"
            
        username = getattr(settings, "RTSP_USERNAME", "")
        password = getattr(settings, "RTSP_PASSWORD", "")
        port = getattr(settings, "RTSP_PORT", 0)
        encoded_pwd = urllib.parse.quote(password)
        return f"rtsp://{username}:{encoded_pwd}@{rtsp_host}:{port}/cam/realmonitor?channel={channel}&subtype=0"

    def start_all(self) -> None:
        """Start all RTSP streams and the consumer thread.

        Uses :class:`StreamManager` to handle each individual stream.
        """
        from .stream_manager import StreamManager
        
        # Re-sync camera_cfg and camera_configs in case they were modified from outside (e.g. in main.py)
        self.camera_cfg = {
            cam_id: cam.get("url") or self.generate_rtsp_url(cam.get("channel", 1))
            for cam_id, cam in self.camera_configs.items()
        }

        from config import settings
        rtsp_password = getattr(settings, "RTSP_PASSWORD", "")
        self._stop_event.clear()

        for cam_id, cam in self.camera_configs.items():
            url = cam.get("url") or self.generate_rtsp_url(cam.get("channel", 1))
            masked_url = url.replace(rtsp_password, "******") if rtsp_password and url != "mock" else url
            logger.info("[CameraManager] Starting stream for %s (Channel %s) with URL: %s", 
                        cam_id, cam.get("channel", "N/A"), masked_url)
            
            stream_mgr = StreamManager(rtsp_url=url, reconnect_interval=self.reconnect_interval, camera_name=cam.get("name", cam_id))
            stream_mgr.start()
            self.streams[cam_id] = stream_mgr
            self._buffers[cam_id] = None

            # Spawn a producer thread that fetches frames from the StreamManager and pushes to the shared queue
            t = threading.Thread(target=self._producer_compat, args=(cam_id, stream_mgr), daemon=True, name=f"{cam_id}-compat-prod")
            t.start()
            self._threads.append(t)

    def start(self) -> None:
        """Start all RTSP streams (alias for start_all)."""
        self.start_all()

    def stop_all(self) -> None:
        """Signal all threads to stop and release resources."""
        self._stop_event.set()
        for s in list(self.streams.values()):
            s.stop()
        self.streams.clear()
        self._threads.clear()

    def _producer_compat(self, cam_id: str, stream_mgr: object) -> None:
        last_ts = None
        while not self._stop_event.is_set():
            latest = stream_mgr.frame_buffer.latest()
            if latest is not None:
                frame, ts = latest
                if ts != last_ts:
                    try:
                        self._queue.put((cam_id, frame, ts), block=False)
                        last_ts = ts
                    except Exception:
                        pass
            time.sleep(0.01)

    # ---------------------------------------------------------------------
    # Internal producer per camera (legacy compatibility)
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
        """Generator yielding ``(camera_id, frame, timestamp)`` tuples."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                yield item
            except Empty:
                continue

    def get_resolution(self, cam_id: str) -> Tuple[int, int]:
        """Return the resolution (width, height) of the latest frame for *cam_id*."""
        stream_mgr = self.streams.get(cam_id)
        if not stream_mgr:
            return (0, 0)
        return stream_mgr.get_resolution()

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def get_active_cameras(self) -> List[Dict[str, any]]:
        """Return a list of camera dictionaries compatible with the legacy code."""
        return list(self.camera_configs.values())

    # ---------------------------------------------------------------------
    # New API delegations used by ``main.py``
    # ---------------------------------------------------------------------
    def get_latest_frame(self, cam_id: str) -> Optional[np.ndarray]:
        """Return the most recent frame for *cam_id* or ``None`` if unavailable."""
        stream_mgr = self.streams.get(cam_id)
        if not stream_mgr:
            return None
        return stream_mgr.get_latest_frame()

    def is_connected(self, cam_id: str) -> bool:
        """Return ``True`` if *cam_id* is currently delivering frames."""
        stream_mgr = self.streams.get(cam_id)
        if not stream_mgr:
            return False
        return stream_mgr.is_connected()

    def get_fps(self, cam_id: str) -> float:
        """Return the measured input FPS for *cam_id* (0.0 if not enough data)."""
        stream_mgr = self.streams.get(cam_id)
        if not stream_mgr:
            return 0.0
        return stream_mgr.get_fps()
