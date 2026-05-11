"""Shared utilities: config loading, logging, seeding, IO helpers."""
from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER_NAME = "age_bias_test"


# ----------------------------- Logging -----------------------------

def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent."""
    logger = logging.getLogger(name or LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(h)
    logger.propagate = False
    return logger


# ----------------------------- Config -----------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in a config tree."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand ${ENV} placeholders."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides like ``foo.bar=42`` to a config dict (in place)."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override (need key=value): {ov!r}")
        key, raw = ov.split("=", 1)
        try:
            value = yaml.safe_load(raw)         # Parses ints/bools/floats.
        except Exception:
            value = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


# ----------------------------- Seeding -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------- Paths / IO -----------------------------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Paths:
    """Resolves all output paths from a config."""
    output_dir: Path
    report_dir: Path

    @property
    def manifests_dir(self) -> Path:
        return ensure_dir(self.output_dir / "manifests")

    @property
    def manipulated_dir(self) -> Path:
        return ensure_dir(self.output_dir / "manipulated")

    @property
    def predictions_dir(self) -> Path:
        return ensure_dir(self.output_dir / "predictions")

    @property
    def metrics_dir(self) -> Path:
        return ensure_dir(self.output_dir / "metrics")

    @classmethod
    def from_config(cls, cfg: dict) -> "Paths":
        exp = cfg["experiment"]
        return cls(
            output_dir=ensure_dir(exp["output_dir"]),
            report_dir=ensure_dir(exp["report_dir"]),
        )
