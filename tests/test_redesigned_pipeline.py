# tests/test_redesigned_pipeline.py
"""Unit tests for the redesigned face recognition, identity, tracking, and attendance managers."""

import unittest
from datetime import datetime, timedelta
import numpy as np
from types import SimpleNamespace

from employee_management.embedding_manager import EmbeddingManager
from session.identity_manager import IdentityManager
from session.track_memory_manager import TrackMemoryManager
from session.attendance_manager import AttendanceManager

class TestRedesignedPipeline(unittest.TestCase):
    
    def setUp(self) -> None:
        self.base_time = datetime(2026, 7, 16, 12, 0, 0)

    def test_embedding_manager_multi_matching(self) -> None:
        mgr = EmbeddingManager(project_root=".")
        # Setup embeddings: EMP001 has two distinct embeddings (e.g. front and side)
        emb_front = np.zeros(512, dtype=np.float32)
        emb_front[0] = 1.0
        emb_side = np.zeros(512, dtype=np.float32)
        emb_side[1] = 1.0
        
        mgr.employee_embeddings = {
            "EMP001": {
                "name": "Arun",
                "embeddings": [emb_front, emb_side],
                "image_count": 2,
                "images_metadata": []
            }
        }
        mgr._build_gallery_matrix()
        
        # Test query matching front
        query_front = np.zeros(512, dtype=np.float32)
        query_front[0] = 0.9
        query_front[1] = 0.1
        query_front = query_front / np.linalg.norm(query_front)
        emp_id, name, sim = mgr.match_embedding(query_front)
        self.assertEqual(emp_id, "EMP001")
        self.assertGreater(sim, 0.8)

        # Test query matching side
        query_side = np.zeros(512, dtype=np.float32)
        query_side[0] = 0.1
        query_side[1] = 0.9
        query_side = query_side / np.linalg.norm(query_side)
        emp_id, name, sim = mgr.match_embedding(query_side)
        self.assertEqual(emp_id, "EMP001")
        self.assertGreater(sim, 0.8)

    def test_identity_manager_consecutive_locking(self) -> None:
        id_mgr = IdentityManager()
        from collections import deque
        track_mem = {
            "camera_id": "CAM001",
            "track_id": 15,
            "locked_status": False,
            "recognition_count": 0,
            "embedding_history": [],
            "last_matched_employee_id": None,
            "consecutive_count": 0,
            "vote_buffer": deque(maxlen=5),
            "employee_id": None,
            "employee_name": "Unknown",
            "recognition_status": "unknown"
        }

        # First match: not locked yet
        newly_locked, locked_id = id_mgr.process_recognition_result(
            track_mem, "EMP001", "Arun", 0.90, 90.0
        )
        self.assertFalse(newly_locked)

        # Second match: not locked yet
        newly_locked, locked_id = id_mgr.process_recognition_result(
            track_mem, "EMP001", "Arun", 0.91, 91.0
        )
        self.assertFalse(newly_locked)

        # Third match: locked! (default settings.MIN_CONSECUTIVE_MATCHES / vote count threshold is 3)
        newly_locked, locked_id = id_mgr.process_recognition_result(
            track_mem, "EMP001", "Arun", 0.89, 89.0
        )
        self.assertTrue(newly_locked)
        self.assertEqual(locked_id, "EMP001")
        self.assertTrue(track_mem["locked_status"])
        self.assertEqual(track_mem["employee_id"], "EMP001")
        self.assertEqual(track_mem["recognition_status"], "identified")

        # Subsequent matching tries to match someone else: should be ignored because track is locked
        newly_locked, locked_id = id_mgr.process_recognition_result(
            track_mem, "EMP002", "Sharma", 0.95, 95.0
        )
        self.assertFalse(newly_locked)
        self.assertEqual(track_mem["employee_id"], "EMP001") # Still EMP001

    def test_track_memory_timeouts(self) -> None:
        mem_mgr = TrackMemoryManager()
        track = mem_mgr.create_track("CAM001", 15, [100, 100, 200, 200], self.base_time)
        
        self.assertEqual(track["track_status"], "tracking")

        # Mark lost
        mem_mgr.mark_lost("CAM001", 15, self.base_time)
        self.assertEqual(track["track_status"], "lost")

        # Process timeouts prior to timeout limit
        exited = mem_mgr.process_timeouts(self.base_time + timedelta(seconds=2), timeout_seconds=5.0)
        self.assertEqual(len(exited), 0)
        self.assertEqual(track["track_status"], "lost")

        # Process timeouts after timeout limit
        exited = mem_mgr.process_timeouts(self.base_time + timedelta(seconds=6), timeout_seconds=5.0)
        self.assertEqual(len(exited), 1)
        self.assertEqual(track["track_status"], "exited")

    def test_attendance_manager_working_hours(self) -> None:
        att_mgr = AttendanceManager(lost_timeout_seconds=5.0)
        session = att_mgr.create_session(
            employee_id="EMP001",
            employee_name="Arun",
            camera_id="CAM001",
            track_id=15,
            bbox=[100, 100, 200, 200],
            timestamp=self.base_time,
            confidence=90.0
        )

        # Update track seen later
        att_mgr.update_track(session, "CAM001", 15, [105, 105, 205, 205], self.base_time + timedelta(seconds=9), [])
        
        # Mark lost
        att_mgr.handle_lost_track("CAM001", 15, self.base_time + timedelta(seconds=9))
        self.assertEqual(session.status, "lost")

        # Process timeouts after timeout threshold
        exited = att_mgr.process_timeouts(self.base_time + timedelta(seconds=15))
        self.assertEqual(len(exited), 1)
        self.assertEqual(session.status, "exited")
        self.assertEqual(session.working_duration, 9.0)

    def test_unrecognized_track_attendance_records(self) -> None:
        import os
        import glob
        from session.attendance_manager import unknown_dir
        
        # Clean existing Unknown attendance files
        pattern = os.path.join(unknown_dir(), "unknown_track_15_*.json")
        for f in glob.glob(pattern):
            try:
                os.unlink(f)
            except OSError:
                pass

        mem_mgr = TrackMemoryManager()
        track = mem_mgr.create_track("CAM001", 15, [100, 100, 200, 200], self.base_time)
        mem_mgr.mark_lost("CAM001", 15, self.base_time + timedelta(seconds=2))
        
        exited = mem_mgr.process_timeouts(self.base_time + timedelta(seconds=8), timeout_seconds=5.0)
        self.assertEqual(len(exited), 1)
        
        att_mgr = AttendanceManager(lost_timeout_seconds=5.0)
        for t in exited:
            if not t["locked_status"]:
                att_mgr.generate_unrecognized_attendance_record(t)

        # Check if record file was generated
        files = glob.glob(pattern)
        self.assertEqual(len(files), 1)
        
        # Cleanup
        for f in files:
            os.unlink(f)

    def test_reid_feature_fusion_and_matching(self) -> None:
        from employee_management.embedding_manager import EmbeddingManager
        import numpy as np
        
        emb_mgr = EmbeddingManager(project_root=".")
        # Setup dummy gallery embeddings
        dummy_feat = np.ones(1280, dtype=np.float32)
        norm = np.linalg.norm(dummy_feat)
        if norm > 0:
            dummy_feat = dummy_feat / norm
        
        emb_mgr.employee_embeddings["EMP001"] = {
            "employee_id": "EMP001",
            "name": "Arun",
            "embeddings": [],
            "reid_embeddings": [dummy_feat],
            "image_count": 1,
            "images_metadata": []
        }
        emb_mgr._build_gallery_matrix()
        
        # Test exact match
        best_id, best_name, score = emb_mgr.match_reid_embedding(dummy_feat)
        self.assertEqual(best_id, "EMP001")
        self.assertEqual(best_name, "Arun")
        self.assertAlmostEqual(score, 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
