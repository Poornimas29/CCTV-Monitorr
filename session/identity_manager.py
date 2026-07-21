# session/identity_manager.py
"""IdentityManager manages Track ID to Employee ID mappings and confirmation locks.

RC-6 fix: Replaced the fragile consecutive-count streak (resets on any bad frame)
with a voting-window approach using the vote_buffer stored in the TrackMemory dict.
Lock condition: 3 of the last 5 recognition results agree on the same employee_id
AND that score was above the RECOGNITION_THRESHOLD.

RC-5 fix: Collision guard now checks for duplicate locks on the SAME camera only.
Cross-camera dual presence is intentionally allowed (employee visible simultaneously
on two cameras that have overlapping coverage on a large production floor).
"""

import logging
from collections import Counter
from typing import Dict, Optional, Tuple, Any
import config.settings as settings

logger = logging.getLogger(__name__)


class IdentityManager:
    """Manages the mapping of Track IDs to Employee IDs, lock states, and vote-based confirmations."""

    def __init__(self) -> None:
        # Maps (camera_id, track_id) -> employee_id  (locked tracks only)
        self.track_to_employee: Dict[Tuple[str, int], str] = {}

    def get_mapped_employee_id(self, camera_id: str, track_id: int) -> Optional[str]:
        """Gets the employee ID mapped to a camera ID and track ID, if locked."""
        return self.track_to_employee.get((camera_id, track_id))

    def process_recognition_result(
        self,
        track: Dict[str, Any],
        matched_emp_id: Optional[str],
        matched_name: str,
        similarity: float,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """Updates the vote buffer and locks identity after sufficient agreement.

        Voting window logic
        -------------------
        Each recognition result adds one vote to a deque(maxlen=5) stored in the
        track dict.  A positive vote (employee_id string) is added when the match
        is above RECOGNITION_THRESHOLD; otherwise None is added.

        Lock condition: the most-voted employee_id has ≥ 3 positive votes in the
        last 5 results.  This means:
          - 5/5 or 4/5 consistent results: locks immediately on the 3rd match.
          - One random bad frame: does NOT reset — the buffer still has the good votes.
          - Three bad frames in a row: would reduce positive votes below 3 and delay
            locking, which is the CORRECT behaviour (bad frames = low quality).

        Returns
        -------
        (is_newly_locked, employee_id_or_none)
        """
        camera_id = track["camera_id"]
        track_id = track["track_id"]

        # If already locked, ignore further attempts
        if track["locked_status"]:
            return False, track["employee_id"]

        # Record this attempt
        track["recognition_count"] += 1
        track["embedding_history"].append(
            {
                "employee_id": matched_emp_id,
                "similarity": similarity,
                "confidence": confidence,
            }
        )

        logger.info(
            "[IdentityManager] Track %d | Camera %s | match=%s | score=%.4f",
            track_id,
            camera_id,
            matched_emp_id or "Unknown",
            similarity,
        )

        # ── Vote ──────────────────────────────────────────────────────────────
        # Add positive vote only if above threshold; None occupies the slot otherwise
        if matched_emp_id and similarity >= settings.RECOGNITION_THRESHOLD:
            track["vote_buffer"].append(matched_emp_id)
        else:
            track["vote_buffer"].append(None)

        # Tally positive votes
        positive_votes = [v for v in track["vote_buffer"] if v is not None]
        if not positive_votes:
            return False, None

        vote_counts = Counter(positive_votes)
        top_emp_id, top_count = vote_counts.most_common(1)[0]

        # Require at least 3 positive agreements
        if top_count < 3:
            return False, None

        # ── Global (same-camera) Collision Guard ──────────────────────────────
        # Prevent two different tracks on the SAME camera from claiming the same
        # employee.  Cross-camera dual-presence is intentionally allowed.
        for (existing_cam, existing_trk), existing_emp_id in self.track_to_employee.items():
            if (
                existing_emp_id == top_emp_id
                and existing_cam == camera_id
                and existing_trk != track_id
            ):
                logger.warning(
                    "[IdentityManager] Lock REJECTED — Track %d tried to claim %s "
                    "but it is already locked to Track %d on Camera %s.",
                    track_id,
                    top_emp_id,
                    existing_trk,
                    camera_id,
                )
                track["vote_buffer"].clear()
                return False, None

        # ── Lock ──────────────────────────────────────────────────────────────
        # Derive the best confidence score from the recent history for this employee
        best_score = max(
            (
                e["similarity"]
                for e in track["embedding_history"]
                if e["employee_id"] == top_emp_id
            ),
            default=similarity,
        )
        best_confidence = round(best_score * 100.0, 1)

        track["locked_status"] = True
        track["employee_id"] = top_emp_id
        track["employee_name"] = matched_name
        track["recognition_status"] = "identified"
        track["recognition_confidence"] = best_confidence

        self.track_to_employee[(camera_id, track_id)] = top_emp_id

        vote_window = len(track["vote_buffer"])
        logger.info(
            "[IdentityManager] LOCKED — Track %d → Employee %s | "
            "votes: %d/%d | confidence: %.1f%%",
            track_id,
            top_emp_id,
            top_count,
            vote_window,
            best_confidence,
        )

        return True, top_emp_id
