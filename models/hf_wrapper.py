"""HuggingFace ``transformers`` age classifier wrapper.

Default model: ``nateraw/vit-age-classifier`` — a ViT-base trained to classify
into 9 age buckets. We map each bucket to its midpoint when computing a point estimate.

Bucket -> years:
  0-2 -> 1
  3-9 -> 6
  10-19 -> 14.5
  20-29 -> 24.5
  30-39 -> 34.5
  40-49 -> 44.5
  50-59 -> 54.5
  60-69 -> 64.5
  more than 70 -> 75
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from models.base import AgeModel, AgePrediction, register
from utils import get_logger

LOG = get_logger(__name__)


# Standard mapping for 9-bucket ViT age classifier.
_BUCKET_MIDPOINTS = {
    "0-2": 1.0, "3-9": 6.0, "10-19": 14.5, "20-29": 24.5,
    "30-39": 34.5, "40-49": 44.5, "50-59": 54.5, "60-69": 64.5,
    "more than 70": 75.0,
}


@register("hf_transformers")
class HFAgeClassifier(AgeModel):
    name = "hf_age_classifier"

    def __init__(self, model_id: str = "nateraw/vit-age-classifier", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._device = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as e:
            raise ImportError("transformers + torch are required.") from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageClassification.from_pretrained(self.model_id).to(self._device)
        self._model.eval()
        LOG.info("Loaded %s on %s", self.model_id, self._device)

    @staticmethod
    def _expected_age(probs: np.ndarray, id2label: dict[int, str]) -> tuple[float, dict[str, float]]:
        """Compute E[age] = sum(p_i * midpoint_i)."""
        dist: dict[str, float] = {}
        exp_age = 0.0
        for i, p in enumerate(probs):
            label = id2label[i]
            mid = _BUCKET_MIDPOINTS.get(label)
            if mid is None:
                # Try to parse "lo-hi" or "more than N".
                if "more than" in label:
                    mid = float(label.split()[-1]) + 5.0
                elif "-" in label:
                    lo, hi = label.split("-")
                    mid = (float(lo) + float(hi)) / 2
                else:
                    mid = float(label)
            dist[label] = float(p)
            exp_age += float(p) * mid
        return exp_age, dist

    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        if self._model is None:
            self.setup()

        import cv2
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        except Exception as e:
            return AgePrediction(age=None, error=f"exception: {type(e).__name__}: {e}")

        id2label = self._model.config.id2label
        exp_age, dist = self._expected_age(probs, id2label)
        return AgePrediction(age=float(exp_age), distribution=dist)
