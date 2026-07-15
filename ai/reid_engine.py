# ai/reid_engine.py
"""FastReID feature extractor module using torchvision.

Extracts normalized feature vectors from person crops using a pre-trained
MobileNetV3 network, with a fallback to HSV torso histograms for crash-proof stability.
"""

import cv2
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Optional

logger = logging.getLogger(__name__)

class FastReIDEngine:
    """Extracts appearance ReID feature embeddings from person crop images."""
    
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_deep_model = False
        
        try:
            # Initialize MobileNetV3 Small as a fast, CPU/GPU friendly feature extractor
            self.model = models.mobilenet_v3_small(weights='DEFAULT')
            self.model.classifier = nn.Identity()  # Remove final classifier layer
            self.model.eval()
            self.model.to(self.device)
            
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.use_deep_model = True
            logger.info("FastReID deep feature extractor loaded successfully on %s.", self.device)
        except Exception as exc:
            logger.warning(
                "Unable to load torchvision model weights. FastReID falling back to HSV color histogram matching. Error: %s",
                exc
            )

    def _compute_appearance_histogram(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Fallback torso HSV color histogram calculation."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
            
        ch, cw = person_crop.shape[:2]
        # Focus on the torso (middle 60% vertically and 80% horizontally)
        ty1, ty2 = int(ch * 0.2), int(ch * 0.8)
        tx1, tx2 = int(cw * 0.1), int(cw * 0.9)
        torso_crop = person_crop[ty1:ty2, tx1:tx2]
        if torso_crop.size == 0:
            return None
            
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def extract_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract appearance features. Returns normalized NumPy array or None."""
        if frame is None or frame.size == 0:
            return None

        if not self.use_deep_model:
            # Fallback to HSV histogram
            return self._compute_appearance_histogram(frame, bbox)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                feat_np = features.squeeze(0).cpu().numpy()
                
                # Normalize features
                norm = np.linalg.norm(feat_np)
                if norm > 0:
                    feat_np = feat_np / norm
                return feat_np
        except Exception as exc:
            logger.error("Failed to extract ReID features: %s. Falling back to HSV histogram.", exc)
            return self._compute_appearance_histogram(frame, bbox)

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute similarity score (cosine similarity or histogram correlation)."""
        if feat1 is None or feat2 is None:
            return 0.0

        if not self.use_deep_model or feat1.ndim > 1 or feat2.ndim > 1:
            # Histogram correlation fallback
            # (ensure shapes match and use correlation comparing method)
            if feat1.shape == feat2.shape:
                return float(cv2.compareHist(feat1, feat2, cv2.HISTCMP_CORREL))
            return 0.0

        # Cosine similarity for normalized vectors
        return float(np.dot(feat1, feat2))
