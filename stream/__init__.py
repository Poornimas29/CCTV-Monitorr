# stream/__init__.py
"""Stream package initialization.

Provides easy imports for the primary stream‑related classes used throughout
the project.  The original code referenced a non‑existent ``RTSPStreamReader``
and ``StreamManager`` which caused an ``ImportError`` when the application
started.  Here we alias the existing ``RTSPStream`` implementation to the
expected name and provide a minimal stub for ``StreamManager`` so that the
import succeeds without altering the rest of the codebase.
"""

from .frame_buffer import FrameBuffer
from .rtsp_stream import RTSPStream as RTSPStreamReader  # Alias for compatibility
from .camera_manager import CameraManager
from .grid_renderer import GridRenderer

# Minimal placeholder to satisfy ``from stream.stream_manager import StreamManager``
# in ``main.py``.  The full implementation is not required for Phase 1, but we
# provide the class so that imports succeed and the module can be extended
# later.

class StreamManager:
    """Placeholder StreamManager.

    In Phase 1 the monitoring logic is orchestrated by ``services.
    monitoring_service.MonitoringService`` which uses ``CameraManager``
    directly.  ``StreamManager`` can be expanded later to include higher‑level
    coordination (e.g., multi‑camera dashboards).  For now it simply stores
    a reference to a ``CameraManager`` instance.
    """

    def __init__(self, camera_manager: CameraManager):
        self.camera_manager = camera_manager

    def start(self):
        self.camera_manager.start()

    def stop(self):
        self.camera_manager.stop()
