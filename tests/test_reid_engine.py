# tests/test_reid_engine.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.reid_engine import FastReIDEngine

class TestReIDEngine(unittest.TestCase):
    
    def test_reid_engine_init(self) -> None:
        engine = FastReIDEngine()
        self.assertIsNotNone(engine.device)

    def test_torso_histogram_fallback_extraction(self) -> None:
        engine = FastReIDEngine()
        # Force fallback mode
        engine.use_deep_model = False
        
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = [10, 10, 90, 90]
        
        hist = engine.extract_features(frame, bbox)
        self.assertIsNotNone(hist)
        # Verify shape of HSV hist (30 * 32 bins)
        self.assertEqual(hist.shape, (30, 32))

    def test_compute_similarity_hist(self) -> None:
        engine = FastReIDEngine()
        engine.use_deep_model = False
        
        hist1 = np.ones((30, 32), dtype=np.float32)
        hist2 = np.ones((30, 32), dtype=np.float32)
        
        similarity = engine.compute_similarity(hist1, hist2)
        self.assertAlmostEqual(similarity, 1.0, places=3)

    def test_compute_similarity_deep(self) -> None:
        engine = FastReIDEngine()
        engine.use_deep_model = True
        
        # Test unit feature vectors
        feat1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feat2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        feat3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.assertAlmostEqual(engine.compute_similarity(feat1, feat2), 1.0)
        self.assertAlmostEqual(engine.compute_similarity(feat1, feat3), 0.0)

if __name__ == "__main__":
    unittest.main()
