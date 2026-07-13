import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.session_engine import EmployeeSessionEngine


class TestEmployeeSessionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = EmployeeSessionEngine(session_timeout_seconds=600)
        self.base_time = datetime(2026, 7, 9, 9, 0, 0)

    def test_first_recognition_creates_session(self) -> None:
        session, event = self.engine.process_recognition(
            employee_id="EMP001",
            employee_name="Rahul",
            confidence=98.4,
            timestamp=self.base_time,
        )

        self.assertEqual(event, "started")
        self.assertEqual(session["employee_id"], "EMP001")
        self.assertEqual(session["employee_name"], "Rahul")
        self.assertEqual(self.engine.active_session_count(), 1)

    def test_same_employee_recognition_updates_existing_session(self) -> None:
        self.engine.process_recognition("EMP001", "Rahul", 98.4, self.base_time)
        session, event = self.engine.process_recognition(
            "EMP001",
            "Rahul",
            97.9,
            self.base_time + timedelta(minutes=10),
        )

        self.assertEqual(event, "updated")
        self.assertEqual(self.engine.active_session_count(), 1)
        self.assertEqual(session["recognition_confidence"], 97.9)
        self.assertEqual(session["last_seen_time"], self.base_time + timedelta(minutes=10))

    def test_multiple_employees_get_independent_sessions(self) -> None:
        self.engine.process_recognition("EMP001", "Rahul", 98.4, self.base_time)
        self.engine.process_recognition("EMP002", "Kumar", 96.7, self.base_time + timedelta(minutes=1))

        self.assertEqual(self.engine.active_session_count(), 2)
        self.assertIn("EMP001", self.engine.get_active_sessions())
        self.assertIn("EMP002", self.engine.get_active_sessions())

    def test_unknown_recognition_does_not_create_session(self) -> None:
        session, event = self.engine.process_recognition(None, "Unknown", 0.0, self.base_time)

        self.assertIsNone(session)
        self.assertEqual(event, "ignored")
        self.assertEqual(self.engine.active_session_count(), 0)


if __name__ == "__main__":
    unittest.main()
