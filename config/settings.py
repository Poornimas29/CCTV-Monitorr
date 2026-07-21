"""
Configuration settings module for the Employee Monitoring System.
Loads environment variables from a .env file and provides typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root — override=True ensures .env always wins even if
# environment variables were cached from a previous run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


# RTSP Stream Settings
RTSP_URL: str = os.getenv("RTSP_URL", "mock")
RTSP_HOST: str = os.getenv("RTSP_HOST", "")
RTSP_USERNAME: str = os.getenv("RTSP_USERNAME", "")
RTSP_PASSWORD: str = os.getenv("RTSP_PASSWORD", "")
try:
    RTSP_PORT: int = int(os.getenv("RTSP_PORT", "0"))
except ValueError:
    RTSP_PORT = 0


# Reconnection interval in seconds
RECONNECT_INTERVAL: int = int(os.getenv("RECONNECT_INTERVAL", "5"))

TARGET_FPS: int = int(os.getenv("TARGET_FPS", "30"))

# Storage Directories
CAPTURE_DIR: str = os.getenv("CAPTURE_DIR", "captures")
LOG_DIR: str = os.getenv("LOG_DIR", "logs")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
LOG_FILE: str = os.path.join(LOG_DIR, "app.log")

# Camera Configuration List
CAMERAS: list[dict] = [
    {
        "id": "CAM002",
        "name": "Working Bay Camera",
        "channel": 6,
        "enabled": True,
        "url": ""
    }
]


# Phase 2 Configs
try:
    CONF_PERSON: float = float(os.getenv("CONF_PERSON", "0.50"))
except ValueError:
    CONF_PERSON = 0.50

try:
    CONF_PHONE: float = float(os.getenv("CONF_PHONE", "0.30"))
except ValueError:
    CONF_PHONE = 0.30

try:
    CONF_UNIFORM: float = float(os.getenv("CONF_UNIFORM", "0.40"))
except ValueError:
    CONF_UNIFORM = 0.40

try:
    CONF_SAFETY_CAP: float = float(os.getenv("CONF_SAFETY_CAP", "0.40"))
except ValueError:
    CONF_SAFETY_CAP = 0.40

try:
    PHONE_USAGE_CONFIRM_SECONDS: float = float(os.getenv("PHONE_USAGE_CONFIRM_SECONDS", "2.0"))
except ValueError:
    PHONE_USAGE_CONFIRM_SECONDS = 2.0

try:
    REID_SIMILARITY_THRESHOLD: float = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.65"))
except ValueError:
    REID_SIMILARITY_THRESHOLD = 0.65

# ─── Face Recognition, Tracking & Session Settings ─────────────────────────
try:
    RECOGNITION_INTERVAL: float = float(os.getenv("RECOGNITION_INTERVAL", "1.0"))
except ValueError:
    RECOGNITION_INTERVAL = 1.0

try:
    RECOGNITION_THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.70"))
except ValueError:
    RECOGNITION_THRESHOLD = 0.70

try:
    MIN_FACE_SIZE: int = int(os.getenv("MIN_FACE_SIZE", "30"))
except ValueError:
    MIN_FACE_SIZE = 30

try:
    MIN_FACE_QUALITY: float = float(os.getenv("MIN_FACE_QUALITY", "3.0"))
except ValueError:
    MIN_FACE_QUALITY = 3.0

try:
    MIN_CONSECUTIVE_MATCHES: int = int(os.getenv("MIN_CONSECUTIVE_MATCHES", "2"))
except ValueError:
    MIN_CONSECUTIVE_MATCHES = 2

try:
    TRACK_TIMEOUT: float = float(os.getenv("TRACK_TIMEOUT", "60.0"))
except ValueError:
    TRACK_TIMEOUT = 60.0

# MAX_RECOGNITION_ATTEMPTS intentionally removed. Recognition retries until
# identity is locked. Permanently giving up on a visible track is a bug, not a feature.

try:
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
except ValueError:
    SIMILARITY_THRESHOLD = 0.50

try:
    UNKNOWN_TRACK_CLEANUP_MINUTES: float = float(os.getenv("UNKNOWN_TRACK_CLEANUP_MINUTES", "15.0"))
except ValueError:
    UNKNOWN_TRACK_CLEANUP_MINUTES = 15.0


def get_playback_url_for_camera(camera_id: str) -> str | None:
    """Return the per-camera playback URL if defined, else None."""
    return os.getenv(f"PLAYBACK_URL_{camera_id.upper()}") or os.getenv("PLAYBACK_URL")


def build_default_rtsp_url(camera_id: str = "") -> str:
    """Construct a full RTSP URL from the individual environment variables.

    If ``RTSP_URL`` is set to something other than the literal ``"mock"`` it is
    returned unchanged.  Otherwise we compose ``rtsp://[user[:pass]@]host:port``.
    Empty username/password parts are omitted cleanly.
    """
    # First, honour an explicit playback override if provided.
    # This lets us feed a pre‑recorded video without touching any live URLs.
    # Set PLAYBACK_URL in the .env; it can be a local file path (e.g. C:/videos/demo.mp4)
    # or an alternative RTSP URL. If ENABLE_PLAYBACK is truthy, we return that URL.
    enable_playback = os.getenv("ENABLE_PLAYBACK", "0") not in ("0", "false", "False")
    playback_url = get_playback_url_for_camera(camera_id)
    if enable_playback and playback_url:
        return playback_url

    # Existing behaviour – honour a concrete RTSP_URL unless it is the literal "mock".
    if RTSP_URL and RTSP_URL.lower() != "mock":
        return RTSP_URL
    auth = ""
    if RTSP_USERNAME and RTSP_PASSWORD:
        auth = f"{RTSP_USERNAME}:{RTSP_PASSWORD}@"
    elif RTSP_USERNAME:
        auth = f"{RTSP_USERNAME}@"
    return f"rtsp://{auth}{RTSP_HOST}:{RTSP_PORT}"
