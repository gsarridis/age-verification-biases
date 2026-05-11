"""MiVOLO age estimation wrapper.

MiVOLO (Multi-Input VOLO) is a SoTA age & gender model from WildChlamydia/MiVOLO.

This wrapper assumes the user has cloned the MiVOLO repo and installed it as a package
(``pip install -e .`` from inside the MiVOLO checkout) and downloaded the checkpoint
they want to use. See https://github.com/WildChlamydia/MiVOLO for details.

If MiVOLO is not installed, ``setup()`` raises a clear ImportError.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.base import AgeModel, AgePrediction, register
from utils import get_logger

LOG = get_logger(__name__)


@register("mivolo")
class MiVOLOAge(AgeModel):
    name = "mivolo"

    def __init__(
        self,
        weights: str,
        detector_weights: Optional[str] = None,
        device: str = "cuda",
        with_persons: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = weights
        self.detector_weights = detector_weights  # YOLO weights for person detection.
        self.device = device
        self.with_persons = with_persons
        self._predictor = None

    def setup(self) -> None:
        try:
            from mivolo.predictor import Predictor
        except ImportError as e:
            raise ImportError(
                "MiVOLO is not installed. Clone https://github.com/WildChlamydia/MiVOLO "
                "and `pip install -e .` inside it, then download a checkpoint."
            ) from e

        # Build a minimal config-like namespace expected by MiVOLO's Predictor.
        class _Args:
            pass

        args = _Args()
        args.checkpoint = self.weights
        args.detector_weights = self.detector_weights
        args.device = self.device
        args.with_persons = self.with_persons
        args.disable_faces = False
        args.draw = False
        # try:
        self._predictor = Predictor(args, verbose=False)
        # except Exception as e:
        #     raise RuntimeError(f"MiVOLO Predictor init failed: {e}") from e

    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        if self._predictor is None:
            self.setup()
        try:
            detected_objects, _ = self._predictor.recognize(image_bgr)
        except Exception as e:
            return AgePrediction(age=None, error=f"exception: {type(e).__name__}: {e}")

        # detected_objects exposes ``ages`` and ``genders`` lists, plus bounding info.
        ages = getattr(detected_objects, "ages", None)
        if not ages:
            return AgePrediction(age=None, error="no_face")

        # If multiple faces are detected, pick the largest face bbox.
        try:
            face_bboxes = detected_objects.face_bboxes
            sizes = [
                (b[2] - b[0]) * (b[3] - b[1]) if b is not None else -1
                for b in face_bboxes
            ]
            idx = int(np.argmax(sizes))
            age = ages[idx]
        except Exception:
            age = ages[0]

        if age is None:
            return AgePrediction(age=None, error="no_age")
        return AgePrediction(age=float(age))
