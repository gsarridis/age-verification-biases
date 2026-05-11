"""InsightFace age estimation wrapper.

InsightFace's ``buffalo_l`` analysis pack ships with a small genderage model
(an MNet-based ONNX). Lightweight, runs on CPU or GPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.base import AgeModel, AgePrediction, register
from utils import get_logger

LOG = get_logger(__name__)


@register("insightface")
class InsightFaceAge(AgeModel):
    name = "insightface_genderage"

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_pack = model_pack
        self.det_size = det_size
        self._app = None

    def setup(self) -> None:
        # try:
        from insightface.app import FaceAnalysis

        # except ImportError as e:
        #     raise ImportError("insightface is required. "
        #                         "Install with: pip install insightface onnxruntime-gpu") from e
        # Use both providers so we get GPU if available, CPU otherwise.
        self._app = FaceAnalysis(
            name=self.model_pack,
            allowed_modules=("detection", "genderage"),
        )
        # ctx_id=0 -> GPU 0; -1 -> CPU. InsightFace handles fallback automatically.
        self._app.prepare(ctx_id=0, det_size=self.det_size)

    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        if self._app is None:
            self.setup()
        try:
            faces = self._app.get(image_bgr)
        except Exception as e:
            return AgePrediction(age=None, error=f"exception: {type(e).__name__}: {e}")
        if not faces:
            return AgePrediction(age=None, error="no_face")
        # Largest face by bbox area.
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        age = getattr(f, "age", None)
        if age is None:
            return AgePrediction(age=None, error="no_age")
        return AgePrediction(
            age=float(age), extra={"gender": int(getattr(f, "gender", -1))}
        )
