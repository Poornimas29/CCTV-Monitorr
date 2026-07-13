"""Employee session tracking for the productivity monitoring workflow.

This module keeps one active session per recognized employee across all
connected cameras. It does not calculate working hours, phone usage, or any
other downstream metrics in this phase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EmployeeSessionEngine:
    """Create and maintain one active session per employee."""

    def __init__(self, session_timeout_seconds: int = 600) -> None:
        self.session_timeout_seconds = session_timeout_seconds
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def process_recognition(
        self,
        employee_id: Optional[str],
        employee_name: Optional[str],
        confidence: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Create or update the active session for a matched employee."""
        if not employee_id or not employee_name:
            return None, "ignored"

        now = timestamp or datetime.now()
        existing = self._active_sessions.get(employee_id)

        if existing and not self._is_session_expired(existing, now):
            existing["last_seen_time"] = now
            existing["recognition_confidence"] = confidence
            existing["status"] = "Present"
            logger.info(
                "Employee Recognized | %s | %s | Confidence: %.1f%% | Existing Session Updated",
                employee_id,
                employee_name,
                confidence,
            )
            print(f"[{now:%H:%M:%S}] Employee Recognized | {employee_id} | {employee_name} | Confidence : {confidence:.1f}% | Existing Session Updated")
            return existing, "updated"

        session = {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "session_start_time": now,
            "last_seen_time": now,
            "status": "Present",
            "recognition_confidence": confidence,
        }
        self._active_sessions[employee_id] = session
        logger.info(
            "Employee Recognized | %s | %s | Confidence: %.1f%% | Session Started",
            employee_id,
            employee_name,
            confidence,
        )
        print(f"[{now:%H:%M:%S}] Employee Recognized | {employee_id} | {employee_name} | Confidence : {confidence:.1f}% | Session Started")
        return session, "started"

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Return the active sessions dictionary."""
        return dict(self._active_sessions)

    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._active_sessions)

    def _is_session_expired(self, session: Dict[str, Any], now: datetime) -> bool:
        last_seen = session.get("last_seen_time")
        if not isinstance(last_seen, datetime):
            return True
        return (now - last_seen) > timedelta(seconds=self.session_timeout_seconds)
