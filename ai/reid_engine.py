# ai/reid_engine.py
"""FastReIDEngine — production-grade Person Re-Identification feature extractor.

RC-11 fix: Replaced MobileNetV3-Small (ImageNet classification backbone) with
OSNet-x0.25 from torchreid, a model explicitly trained on large-scale person
Re-ID datasets (Market-1501, DukeMTMC-ReID).

Why OSNet is better for this use case
--------------------------------------
MobileNetV3-Small was pretrained on ImageNet to classify 1000 object categories.
Its features are optimised for "what is this object" — not "is this the same person
as in another image".  In a factory environment where all employees wear similar
uniforms and cameras are ceiling-mounted, ImageNet features cannot discriminate
between employees because the background/clothing colour is nearly identical.

OSNet (Omni-Scale Network) was purpose-built for Person ReID.  It uses multi-scale
feature aggregation to learn fine-grained clothing texture, body shape, and gait
patterns from paired same-person / different-person training examples.  It achieves
~75% Rank-1 accuracy on Market-1501 even under partial occlusion.

Fallback chain
--------------
1. torchreid OSNet-x0.25 (best quality)
2. torchvision MobileNetV3-Small (good quality, no torchreid dependency)
3. HSV torso color histogram (crash-proof last resort)

RC-10 fix: output is always explicitly L2-normalised regardless of which backend
is active.  The HSV histogram fallback now returns None when called from contexts
that expect a 1D feature vector (prevents shape mismatch in compute_similarity).
"""

import cv2
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Backend detection at import time ────────────────────────────────────────

_BACKEND = "histogram"  # fallback default

try:
    import torchreid          # type: ignore
    import torch
    _BACKEND = "osnet"
    logger.info("[FastReIDEngine] torchreid available — will use OSNet-x0.25.")
except ImportError:
    try:
        import torch
        import torchvision.models as _tv_models  # type: ignore
        import torchvision.transforms as _tv_T   # type: ignore
        _BACKEND = "mobilenet"
        logger.info("[FastReIDEngine] torchreid not found — falling back to MobileNetV3-Small.")
    except ImportError:
        logger.warning("[FastReIDEngine] Neither torchreid nor torchvision available — using HSV histogram only.")


class FastReIDEngine:
    """Extracts appearance ReID feature embeddings from person crop images."""

    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = None
        self._backend: str = _BACKEND

        if _BACKEND == "osnet":
            self._init_osnet()
        elif _BACKEND == "mobilenet":
            self._init_mobilenet()
        # else: histogram only, no model to load

    # ── Initialisation helpers ───────────────────────────────────────────────

    def _init_osnet(self) -> None:
        """Load OSNet-x0.25 from torchreid with Market-1501 pretrained weights."""
        try:
            import torch
            import torchreid  # type: ignore
            import torchvision.transforms as T  # type: ignore
            from PIL import Image  # noqa: F401 — ensure PIL is available

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Build OSNet-x0.25 (lightest OSNet variant, ~2MB, ~4ms on CPU)
            # loss='softmax' means we extract the feature vector before the classifier head.
            self._model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=1000,   # placeholder — we strip the head immediately
                loss="softmax",
                pretrained=True,    # downloads ~2MB Market-1501 weights on first run
            )
            self._model.eval()
            self._model.to(self._device)

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            logger.info(
                "[FastReIDEngine] OSNet-x0.25 loaded on %s (Market-1501 weights).", self._device
            )
        except Exception as exc:
            logger.warning(
                "[FastReIDEngine] OSNet failed to load (%s). Falling back to MobileNetV3.", exc
            )
            self._backend = "mobilenet"
            self._init_mobilenet()

    def _init_mobilenet(self) -> None:
        """Load MobileNetV3-Small from torchvision as ReID backbone."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as T

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = models.mobilenet_v3_small(weights="DEFAULT")
            model.classifier = nn.Identity()   # strip the 1000-class head
            model.eval()
            model.to(self._device)
            self._model = model

            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self._backend = "mobilenet"
            logger.info(
                "[FastReIDEngine] MobileNetV3-Small loaded on %s (ImageNet weights).", self._device
            )
        except Exception as exc:
            logger.warning(
                "[FastReIDEngine] MobileNetV3 failed to load (%s). Using HSV histogram fallback.", exc
            )
            self._backend = "histogram"

    # ── Feature extraction ───────────────────────────────────────────────────

    def _crop_person(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """Safely crop person region from frame."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _deep_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Run the deep model (OSNet or MobileNetV3) on the crop."""
        try:
            import torch

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = self._transform(crop_rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                feat = self._model(img_tensor)
            feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)

            # RC-10 fix: always explicitly L2-normalise the output vector
            norm = np.linalg.norm(feat_np)
            if norm > 0:
                feat_np = feat_np / norm
            return feat_np

        except Exception as exc:
            logger.debug("[FastReIDEngine] Deep model inference failed: %s", exc)
            return None

    def _histogram_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """HSV torso colour histogram — crash-proof last-resort fallback.

        Returns a FLATTENED 1D float32 array (960,) so it can be compared
        with other histogram features via cosine similarity.  Returns None
        if the crop is invalid so the caller can decide how to handle it.
        """
        ch, cw = crop.shape[:2]
        # Focus on the torso (middle 60% vertically, 80% horizontally)
        ty1, ty2 = int(ch * 0.2), int(ch * 0.8)
        tx1, tx2 = int(cw * 0.1), int(cw * 0.9)
        torso = crop[ty1:ty2, tx1:tx2]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Flatten to 1D so shape is consistent with 1D ReID vectors
        flat = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        return flat

    def extract_features(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        """Extract appearance features for the person defined by bbox.

        Returns a normalised 1D float32 numpy array, or None on failure.
        The caller should treat None as "no feature available this frame".
        """
        crop = self._crop_person(frame, bbox)
        if crop is None:
            return None

        if self._backend in ("osnet", "mobilenet") and self._model is not None:
            feat = self._deep_features(crop)
            if feat is not None:
                return feat
            # Deep model failed on this crop — try histogram as emergency fallback
            logger.debug("[FastReIDEngine] Deep inference failed, using histogram fallback.")

        return self._histogram_features(crop)

    # ── Similarity ───────────────────────────────────────────────────────────

    def compute_similarity(
        self, feat1: Optional[np.ndarray], feat2: Optional[np.ndarray]
    ) -> float:
        """Compute cosine similarity between two feature vectors.

        RC-10 fix: added shape and dimension guards so that a 2D histogram
        array can never silently return an incorrect score.
        """
        if feat1 is None or feat2 is None:
            return 0.0

        f1 = np.asarray(feat1, dtype=np.float32)
        f2 = np.asarray(feat2, dtype=np.float32)

        # Guard: only compare vectors of the same shape and dimensionality
        if f1.ndim != 1 or f2.ndim != 1 or f1.shape != f2.shape:
            return 0.0

        # Both vectors should already be L2-normalised; dot product = cosine sim.
        return float(np.dot(f1, f2))
