"""DeepFace age model wrapper.

DeepFace's ``Age`` model is a VGG-based regressor producing an integer age.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from models.base import AgeModel, AgePrediction, register
from utils import get_logger

LOG = get_logger(__name__)


@register("deepface")
class DeepFaceAge(AgeModel):
    name = "deepface_age"

    def __init__(self, detector_backend: str = "opencv", **kwargs):
        super().__init__(**kwargs)
        self.detector_backend = detector_backend
        self._df: Optional[object] = None

    def setup(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise ImportError("deepface is required. Install with: pip install deepface") from e
        # Trigger lazy weight download so failures happen here, not mid-batch.
        self._df = DeepFace
        try:
            self._df.build_model("Age")
        except Exception as e:
            LOG.warning("DeepFace.build_model('Age') warned: %s", e)

    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        if self._df is None:
            self.setup()

        try:
            results = self._df.analyze(
                img_path=image_bgr,                 # DeepFace accepts numpy BGR.
                actions=("age",),
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )
        except Exception as e:
            return AgePrediction(age=None, error=f"exception: {type(e).__name__}: {e}")

        # DeepFace returns a list of dicts (one per detected face).
        if not results:
            return AgePrediction(age=None, error="no_face")
        if isinstance(results, dict):
            results = [results]

        # Take the largest face if multiple.
        def area(r):
            r = r.get("region", {})
            return r.get("w", 0) * r.get("h", 0)
        best = max(results, key=area)
        age = best.get("age")
        if age is None:
            return AgePrediction(age=None, error="no_age")
        return AgePrediction(age=float(age))
