# tracking/byte_tracker.py
"""ByteTrack with Kalman filter motion prediction.

The original IoU-only ByteTracker loses tracks whenever a person moves quickly
between frames (running, bending, turning) because the overlap between the
previous bbox and the current detection drops below 30%.

This version adds a constant-velocity Kalman filter to each track. Before each
matching cycle the filter predicts the next position from velocity history,
dramatically increasing the IoU between prediction and detection even under
fast motion.

State vector: [cx, cy, w, h, vx, vy]
    cx, cy  — center of bounding box
    w, h    — width and height
    vx, vy  — horizontal and vertical velocity (pixels/frame)

Measurement vector: [cx, cy, w, h]
"""

import numpy as np
from typing import List, Tuple


# ── Kalman Filter ────────────────────────────────────────────────────────────

class KalmanBoxFilter:
    """Constant-velocity Kalman filter for a single bounding box track."""

    # Process noise: how much we trust the motion model
    _Q_VAR_POS = 1.0        # position process noise variance
    _Q_VAR_VEL = 10.0       # velocity process noise variance

    # Measurement noise: how much we trust the detection bbox
    _R_VAR = 4.0

    def __init__(self, bbox_tlbr: np.ndarray) -> None:
        """Initialise from a [x1, y1, x2, y2] bounding box."""
        cx, cy, w, h = self._tlbr_to_cxcywh(bbox_tlbr)

        # State: [cx, cy, w, h, vx, vy]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = 1.0   # cx += vx
        self.F[1, 5] = 1.0   # cy += vy

        # Measurement matrix — we observe [cx, cy, w, h]
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0   # cx
        self.H[1, 1] = 1.0   # cy
        self.H[2, 2] = 1.0   # w
        self.H[3, 3] = 1.0   # h

        # Process noise covariance Q
        self.Q = np.diag([
            self._Q_VAR_POS, self._Q_VAR_POS,
            self._Q_VAR_POS, self._Q_VAR_POS,
            self._Q_VAR_VEL, self._Q_VAR_VEL
        ])

        # Measurement noise covariance R
        self.R = np.eye(4, dtype=np.float64) * self._R_VAR

        # Error covariance — start with high uncertainty in velocity
        self.P = np.eye(6, dtype=np.float64)
        self.P[4, 4] = 1000.0
        self.P[5, 5] = 1000.0

    # ── Core KF steps ────────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Predict the next state. Returns predicted [x1, y1, x2, y2]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._cxcywh_to_tlbr(self.x[:4])

    def update(self, bbox_tlbr: np.ndarray) -> np.ndarray:
        """Update state with new observed [x1, y1, x2, y2]. Returns corrected bbox."""
        z = self._tlbr_to_cxcywh(bbox_tlbr)
        y = z - self.H @ self.x                             # Innovation
        S = self.H @ self.P @ self.H.T + self.R             # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)           # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self._cxcywh_to_tlbr(self.x[:4])

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tlbr_to_cxcywh(tlbr: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = tlbr
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float64)

    @staticmethod
    def _cxcywh_to_tlbr(cxcywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = cxcywh
        w, h = max(1.0, w), max(1.0, h)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    @property
    def predicted_tlbr(self) -> np.ndarray:
        """Return the last predicted bbox without running another prediction step."""
        cx, cy, w, h = self.x[:4]
        w, h = max(1.0, w), max(1.0, h)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


# ── STrack (Track representation) ───────────────────────────────────────────

class STrack:
    """Track representation: ID, observed bbox, Kalman-predicted bbox, score, lost count."""

    def __init__(self, tlbr: np.ndarray, score: float, track_id: int) -> None:
        self.tlbr = np.asarray(tlbr, dtype=np.float32)    # Last *observed* bbox
        self.score = score
        self.track_id = track_id
        self.lost_count = 0
        self.kalman = KalmanBoxFilter(self.tlbr)

    @property
    def predicted_tlbr(self) -> np.ndarray:
        """Predicted bbox from Kalman filter (used for IoU matching)."""
        return self.kalman.predicted_tlbr


# ── IoU helper ───────────────────────────────────────────────────────────────

def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU of two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


# ── ByteTracker ──────────────────────────────────────────────────────────────

class ByteTracker:
    """Kalman-augmented ByteTrack multi-object tracker.

    Matching order:
        1. Active tracks × high-confidence detections (IoU on PREDICTED bbox)
        2. Unmatched tracks × low-confidence detections  (IoU on PREDICTED bbox)
        3. Initialise new tracks from leftover high-confidence detections
        4. Increment lost_count on still-unmatched tracks; purge at max_lost
    """

    def __init__(self, max_lost: int = 30, track_thresh: float = 0.5) -> None:
        self.max_lost = max_lost
        self.track_thresh = track_thresh
        self.next_id = 1
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []

    def update(self, dets: np.ndarray) -> List[STrack]:
        """Update tracker with new detections.

        Parameters
        ----------
        dets : np.ndarray
            Shape (N, 5): [x1, y1, x2, y2, score] for each detection.

        Returns
        -------
        List[STrack]
            Currently *visible* (lost_count == 0) tracks.
        """
        # ── 0. Predict next position for all existing tracks ─────────────────
        for trk in self.tracked_stracks + self.lost_stracks:
            trk.kalman.predict()

        # ── 1. Split detections into high/low confidence ─────────────────────
        high_dets: List[np.ndarray] = []
        low_dets: List[np.ndarray] = []
        for det in dets:
            score = float(det[4])
            if score >= self.track_thresh:
                high_dets.append(det)
            elif score >= 0.1:
                low_dets.append(det)

        # ── 2. Match active + lost tracks against high-confidence detections ──
        all_tracks = self.tracked_stracks + self.lost_stracks
        matched_tracks, unmatched_tracks, used_high = self._match(all_tracks, high_dets, iou_thresh=0.3)

        # ── 3. Match still-unmatched tracks against low-confidence detections ─
        matched_low, still_unmatched, _ = self._match(unmatched_tracks, low_dets, iou_thresh=0.3)
        matched_tracks.extend(matched_low)

        # ── 4. Update matched tracks ─────────────────────────────────────────
        new_tracked: List[STrack] = []
        for trk, det in matched_tracks:
            trk.tlbr = trk.kalman.update(det[:4])     # Kalman correction step
            trk.score = float(det[4])
            trk.lost_count = 0
            new_tracked.append(trk)

        # ── 5. Increment lost count; keep until max_lost is exceeded ──────────
        new_lost: List[STrack] = []
        for trk in still_unmatched:
            trk.lost_count += 1
            if trk.lost_count <= self.max_lost:
                new_lost.append(trk)

        # ── 6. Initialise new tracks from unmatched high-confidence detections ─
        for det in (d for i, d in enumerate(high_dets) if i not in used_high):
            new_trk = STrack(det[:4], float(det[4]), self.next_id)
            self.next_id += 1
            new_tracked.append(new_trk)

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost

        # Return only currently visible tracks
        return [t for t in self.tracked_stracks if t.lost_count == 0]

    # ── Matching helper ───────────────────────────────────────────────────────

    @staticmethod
    def _match(
        tracks: List[STrack],
        dets: List[np.ndarray],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[STrack, np.ndarray]], List[STrack], set]:
        """Greedy IoU matching using Kalman-predicted bboxes.

        Returns
        -------
        matched       : list of (track, detection) pairs
        unmatched_trk : tracks that found no match
        used_det_idx  : set of detection indices that were consumed
        """
        matched: List[Tuple[STrack, np.ndarray]] = []
        unmatched_trk: List[STrack] = []
        used_det_idx: set = set()

        for trk in tracks:
            best_iou = 0.0
            best_idx = -1
            pred_box = trk.predicted_tlbr          # Use Kalman prediction

            for idx, det in enumerate(dets):
                if idx in used_det_idx:
                    continue
                iou = bbox_iou(pred_box, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1 and best_iou >= iou_thresh:
                used_det_idx.add(best_idx)
                matched.append((trk, dets[best_idx]))
            else:
                unmatched_trk.append(trk)

        return matched, unmatched_trk, used_det_idx
