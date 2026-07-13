"""
Reconnection handler module for managing connection status and retry intervals.
Provides thread-safe retry counting and state management.
"""

import time
import logging

logger = logging.getLogger(__name__)


class ReconnectHandler:
    """
    Handles state tracking and backoff logic for camera reconnect attempts.
    """
    def __init__(self, reconnect_interval: int = 5) -> None:
        """
        Initializes the reconnection handler.

        Args:
            reconnect_interval: Wait time in seconds between connection retries.
        """
        self.reconnect_interval: int = reconnect_interval
        self.attempts: int = 0
        self.is_reconnecting: bool = False

    def start_reconnect(self) -> None:
        """Logs the start of a reconnection phase."""
        if not self.is_reconnecting:
            self.is_reconnecting = True
            logger.warning("Disconnected - Starting automatic reconnection sequence.")

    def wait_and_retry(self) -> None:
        """
        Increments the attempt counter, logs the attempt, and sleeps
        for the configured reconnect interval.
        """
        self.attempts += 1
        logger.info(
            f"Reconnect Attempt: Trying to reconnect (Attempt #{self.attempts}) "
            f"in {self.reconnect_interval} seconds..."
        )
        time.sleep(self.reconnect_interval)

    def reset(self) -> None:
        """Resets reconnection state upon successful connection."""
        if self.attempts > 0 or self.is_reconnecting:
            logger.info(f"Connected Successfully after {self.attempts} reconnect attempt(s).")
        self.attempts = 0
        self.is_reconnecting = False
