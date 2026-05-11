"""FairFace age model wrapper.

FairFace (https://github.com/joojs/fairface) is a ResNet34 trained for fairness across
race and gender. It outputs age as a 9-bucket classifier with the same buckets as
``nateraw/vit-age-classifier``.

This wrapper assumes you have downloaded one of the FairFace checkpoints (``res34_fair_align_multi_4_20190809.pt``
or the 7-race variant) and pass its path via ``weights``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from models.base import AgeModel, AgePrediction, register
from utils import get_logger

LOG = get_logger(__name__)


_BUCKET_MIDPOINTS = [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0]
_BUCKET_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39",
                  "40-49", "50-59", "60-69", "70+"]


@register("fairface")
class FairFaceAge(AgeModel):
    name = "fairface"

    # FairFace has two variants: 4-race (18 outputs) and 7-race (18 or 18+1 outputs)
    # depending on the checkpoint. We focus on the *age* head, which is the last 9 logits
    # in both released checkpoints. For the 4-race model, outputs are:
    #   [race(4), gender(2), age(9)] => 15 in some checkpoints,
    #   or [gender(2), age(9), race(7)] => 18 in others.
    # We let the user override AGE_SLICE if needed.
    age_slice: tuple[int, int] = (-9, None)   # Default: last 9.

    def __init__(self, weights: str, age_slice: Optional[tuple[int, int]] = None,
                 device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self.device = device
        self._model = None
        self._transform = None
        if age_slice is not None:
            self.age_slice = age_slice

    def setup(self) -> None:
        try:
            import torch
            import torchvision
            from torchvision import transforms
        except ImportError as e:
            raise ImportError("torch + torchvision are required.") from e

        device = self.device if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        self.device = device

        # FairFace ships a ResNet34 architecture trained from scratch.
        model = torchvision.models.resnet34(weights=None)
        # The released checkpoints use 18 output classes; load weights and let it adapt.
        # We probe the checkpoint to determine the FC size.
        ckpt = torch.load(self.weights, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        # Find the fc weight tensor.
        fc_w = ckpt.get("fc.weight")
        if fc_w is None:
            raise RuntimeError(f"Could not find fc.weight in {self.weights}; "
                               "is this a FairFace checkpoint?")
        n_out = fc_w.shape[0]
        model.fc = torch.nn.Linear(model.fc.in_features, n_out)
        model.load_state_dict(ckpt, strict=True)
        model = model.to(device).eval()
        self._model = model

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        LOG.info("Loaded FairFace from %s (out=%d) on %s", self.weights, n_out, device)

    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        import cv2
        import torch

        if self._model is None:
            self.setup()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            x = self._transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self._model(x)[0].cpu().numpy()
        except Exception as e:
            return AgePrediction(age=None, error=f"exception: {type(e).__name__}: {e}")

        s, e = self.age_slice
        age_logits = logits[s:e] if e is not None else logits[s:]
        # Softmax -> midpoint expectation.
        probs = np.exp(age_logits - age_logits.max())
        probs = probs / probs.sum()
        if len(probs) != len(_BUCKET_MIDPOINTS):
            return AgePrediction(age=None,
                                 error=f"unexpected age head size {len(probs)}")
        exp_age = float(np.dot(probs, _BUCKET_MIDPOINTS))
        dist = {lbl: float(p) for lbl, p in zip(_BUCKET_LABELS, probs)}
        return AgePrediction(age=exp_age, distribution=dist)
