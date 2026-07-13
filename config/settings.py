"""
Configuration settings module for the Employee Monitoring System.
Loads environment variables from a .env file and provides typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# RTSP Stream Settings
RTSP_URL: str = os.getenv("RTSP_URL", "mock")
RTSP_HOST: str = os.getenv("RTSP_HOST", "")
RTSP_USERNAME: str = os.getenv("RTSP_USERNAME", "")
RTSP_PASSWORD: str = os.getenv("RTSP_PASSWORD", "")
try:
    RTSP_PORT: int = int(os.getenv("RTSP_PORT", "0"))
except ValueError:
    RTSP_PORT = 0




# Camera Configuration List
CAMERAS: list[dict] = [
    {
        "id": "CAM001",
        "name": "Billing Camera",
        "channel": 3,
        "enabled": True
    },
    {
        "id": "CAM002",
        "name": "Working Bay Camera",
        "channel": 6,
        "enabled": True
    },
    {
        "id": "CAM003",
        "name": "Packing Area",
        "channel": 4,
        "enabled": True
    },{
        "id": "CAM004",
        "name": "Hall Area",
        "channel": 10,
        "enabled": True
    }
]

# Reconnection interval in seconds
try:
    RECONNECT_INTERVAL: int = int(os.getenv("RECONNECT_INTERVAL", "5"))
except ValueError:
    RECONNECT_INTERVAL = 5

# Target Frames Per Second for display thread
try:
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "25"))
except ValueError:
    TARGET_FPS = 25

# Storage Directories
CAPTURE_DIR: str = os.getenv("CAPTURE_DIR", "captures")
LOG_DIR: str = os.getenv("LOG_DIR", "logs")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")

# Log file path
LOG_FILE: str = os.path.join(LOG_DIR, "app.log")

def build_default_rtsp_url() -> str:
    """Construct a full RTSP URL from the individual environment variables.

    If ``RTSP_URL`` is set to something other than the literal ``"mock"`` it is
    returned unchanged.  Otherwise we compose ``rtsp://[user[:pass]@]host:port``.
    Empty username/password parts are omitted cleanly.
    """
    if RTSP_URL and RTSP_URL.lower() != "mock":
        return RTSP_URL
    auth = ""
    if RTSP_USERNAME and RTSP_PASSWORD:
        auth = f"{RTSP_USERNAME}:{RTSP_PASSWORD}@"
    elif RTSP_USERNAME:
        auth = f"{RTSP_USERNAME}@"
    return f"rtsp://{auth}{RTSP_HOST}:{RTSP_PORT}"
