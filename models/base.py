"""Common interface for age estimation models.

Every model wrapper inherits from ``AgeModel`` and implements ``predict_age(image_bgr)``.

Predictions return a dataclass with:
  * point estimate (years),
  * (optional) full age distribution if the model provides one,
  * an error code if the model failed (e.g., no face detected).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AgePrediction:
    age: Optional[float]                                  # Point estimate in years.
    distribution: Optional[dict[str, float]] = None       # Optional bucketed probs.
    error: Optional[str] = None                           # "no_face", "exception", etc.
    extra: dict = field(default_factory=dict)             # Any model-specific debug info.


class AgeModel(ABC):
    """Base class for all age estimation models."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def setup(self) -> None:
        """Optional: load weights, warm up the GPU, etc. Called once before any prediction."""

    @abstractmethod
    def predict_age(self, image_bgr: np.ndarray) -> AgePrediction:
        """Predict the age in years for a single BGR image."""

    def predict_batch(self, images_bgr: list[np.ndarray]) -> list[AgePrediction]:
        """Default batch implementation: serial. Override for true batching."""
        return [self.predict_age(im) for im in images_bgr]

    def teardown(self) -> None:
        """Optional: release GPU memory."""


# ----- Registry -----

_REGISTRY: dict[str, type[AgeModel]] = {}


def register(backend_name: str):
    """Decorator: register an AgeModel subclass under a backend identifier."""
    def deco(cls: type[AgeModel]) -> type[AgeModel]:
        if backend_name in _REGISTRY:
            raise ValueError(f"Backend '{backend_name}' already registered.")
        _REGISTRY[backend_name] = cls
        return cls
    return deco


def build_model(backend: str, **kwargs) -> AgeModel:
    if backend not in _REGISTRY:
        raise KeyError(f"Unknown model backend '{backend}'. "
                       f"Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[backend](**kwargs)


def list_backends() -> list[str]:
    return sorted(_REGISTRY)
