"""
Logging configuration module for setting up application-wide logger.
Writes to both the console and a file inside the log directory.
"""

import os
import logging
from config.settings import LOG_DIR, LOG_FILE


def setup_logging() -> None:
    """
    Configures standard logging.
    Ensures log directory exists, and routes messages to logs/app.log and console.
    """
    try:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")

    # Set up basic configuration for root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configuration initialized.")
