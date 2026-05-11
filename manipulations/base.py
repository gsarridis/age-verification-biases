"""Common interface for image manipulations.

A *manipulation* takes an input image (and optionally pre-computed face landmarks)
and returns a modified image. All manipulations are deterministic given the same input
and a fixed seed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceContext:
    """Pre-computed face information for an image, shared across manipulations.

    Computing landmarks once per image (rather than once per manipulation) is a major
    speedup when applying many manipulations to the same sample.
    """
    landmarks: np.ndarray | None        # (N, 2) array in pixel coords, or None.
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2), or None.
    image_shape: tuple[int, int]        # (H, W)


class Manipulation(ABC):
    """A single, named, deterministic image manipulation."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        """Return a manipulated copy of ``image`` (BGR, uint8)."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Manipulation {self.name}>"


# ----- Registry -----

_REGISTRY: dict[str, type[Manipulation]] = {}


def register(cls: type[Manipulation]) -> type[Manipulation]:
    """Class decorator: registers a Manipulation subclass under its ``name`` attribute."""
    if not cls.name or cls.name == "base":
        raise ValueError(f"{cls.__name__} must set a non-empty 'name' class attribute.")
    if cls.name in _REGISTRY:
        raise ValueError(f"Manipulation '{cls.name}' is already registered.")
    _REGISTRY[cls.name] = cls
    return cls


def get_manipulation(name: str, **kwargs) -> Manipulation:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown manipulation '{name}'. "
                       f"Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_manipulations() -> list[str]:
    return sorted(_REGISTRY)
