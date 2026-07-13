# tracking/byte_tracker.py
"""Pure-Python implementation of ByteTrack matching logic.

This file provides the STrack and ByteTracker classes to keep track of bounding
boxes across frames using greedy Intersection over Union (IoU) matching.
"""

import numpy as np
from typing import List

class STrack:
    """Track representation containing track ID, bounding box, score, and lost count."""
    def __init__(self, tlbr: np.ndarray, score: float, track_id: int):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = score
        self.track_id = track_id
        self.lost_count = 0


def bbox_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class ByteTracker:
    """Lightweight pure-Python multi-object tracker."""
    def __init__(self, max_lost: int = 30, track_thresh: float = 0.5):
        self.max_lost = max_lost
        self.track_thresh = track_thresh
        self.next_id = 1
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []

    def update(self, dets: np.ndarray) -> List[STrack]:
        """Update the tracker with new detections.

        Parameters
        ----------
        dets: np.ndarray
            NumPy array of shape (N, 5) where each row is [x1, y1, x2, y2, score].

        Returns
        -------
        List[STrack]
            Currently active/visible tracks.
        """
        # 1. Split detections into high score and low score
        high_dets = []
        low_dets = []
        for det in dets:
            score = det[4]
            if score >= self.track_thresh:
                high_dets.append(det)
            elif score >= 0.1:
                low_dets.append(det)
                
        # 2. Match active tracks with high-score detections using greedy IoU matching
        matched_tracks = []
        unmatched_tracks = []
        
        all_tracks = self.tracked_stracks + self.lost_stracks
        used_det_indices = set()
        
        for trk in all_tracks:
            best_iou = 0.0
            best_idx = -1
            for idx, det in enumerate(high_dets):
                if idx in used_det_indices:
                    continue
                iou = bbox_iou(trk.tlbr, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx != -1 and best_iou >= 0.3:
                used_det_indices.add(best_idx)
                trk.tlbr = np.asarray(high_dets[best_idx][:4])
                trk.score = high_dets[best_idx][4]
                trk.lost_count = 0
                matched_tracks.append(trk)
            else:
                unmatched_tracks.append(trk)
                
        unmatched_high_dets = [det for idx, det in enumerate(high_dets) if idx not in used_det_indices]
        
        # 3. Match unmatched tracks with low-score detections
        still_unmatched_tracks = []
        used_low_det_indices = set()
        
        for trk in unmatched_tracks:
            best_iou = 0.0
            best_idx = -1
            for idx, det in enumerate(low_dets):
                if idx in used_low_det_indices:
                    continue
                iou = bbox_iou(trk.tlbr, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx != -1 and best_iou >= 0.3:
                used_low_det_indices.add(best_idx)
                trk.tlbr = np.asarray(low_dets[best_idx][:4])
                trk.score = low_dets[best_idx][4]
                trk.lost_count = 0
                matched_tracks.append(trk)
            else:
                still_unmatched_tracks.append(trk)
                
        # 4. Handle still unmatched tracks (increment lost count)
        new_tracked = []
        new_lost = []
        
        for trk in matched_tracks:
            new_tracked.append(trk)
            
        for trk in still_unmatched_tracks:
            trk.lost_count += 1
            if trk.lost_count <= self.max_lost:
                new_lost.append(trk)
                
        # 5. Initialize new tracks from unmatched high-score detections
        for det in unmatched_high_dets:
            new_trk = STrack(det[:4], det[4], self.next_id)
            self.next_id += 1
            new_tracked.append(new_trk)
            
        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost
        
        # Return only currently active tracked stracks
        return [t for t in self.tracked_stracks if t.lost_count == 0]
