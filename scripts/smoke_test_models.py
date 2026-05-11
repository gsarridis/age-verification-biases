"""Smoke-test each enabled model on a single image.

Run this on your machine after installing dependencies and (where applicable)
downloading model weights. It loads each enabled model in turn, runs one
prediction on a test image, and prints the results. Failures don't stop the
script — it tells you which models work and which don't.

Usage:
    # Use any face image. The astronaut from scikit-image works fine.
    python -m scripts.smoke_test_models --image /path/to/face.jpg

    # Or omit --image to use scikit-image's astronaut sample.
    python -m scripts.smoke_test_models
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._args import add_common_args
from utils import apply_overrides, load_config

# Importing each wrapper registers its backend.
import models.deepface_wrapper  # noqa: F401
import models.insightface_wrapper  # noqa: F401
import models.hf_wrapper  # noqa: F401
import models.mivolo_wrapper  # noqa: F401
import models.fairface_wrapper  # noqa: F401
from models.base import build_model


def _astronaut() -> "any":
    """Fall back to scikit-image's astronaut as a default test image."""
    try:
        from skimage.data import astronaut
    except ImportError:
        return None
    return cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test each model wrapper.")
    add_common_args(parser)
    parser.add_argument("--image", default=None,
                        help="Path to a face image. Defaults to scikit-image's astronaut.")
    args = parser.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: could not read {args.image}")
            return 1
    else:
        img = _astronaut()
        if img is None:
            print("ERROR: no --image given and scikit-image not installed.")
            return 1

    print(f"Testing on image of shape {img.shape}\n")
    print("=" * 78)

    summary: list[tuple[str, str, str]] = []      # (model_name, status, detail)

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        backend = model_cfg["backend"]
        enabled = model_cfg.get("enabled", True)
        kwargs = {k: v for k, v in model_cfg.items()
                  if k not in ("name", "backend", "enabled")}

        status_line = f"[{name:<30s} backend={backend:<16s}]"
        if not enabled:
            print(f"{status_line} SKIPPED (disabled in config)")
            summary.append((name, "skipped", "disabled in config"))
            continue

        print(f"{status_line} loading…", flush=True)
        try:
            t0 = time.time()
            m = build_model(backend, **kwargs)
            m.setup()
            t_load = time.time() - t0
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  FAILED to load — {msg}")
            summary.append((name, "load_failed", msg))
            continue

        try:
            t0 = time.time()
            pred = m.predict_age(img)
            t_pred = time.time() - t0
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  FAILED at predict — {msg}")
            summary.append((name, "predict_failed", msg))
            continue

        if pred.error:
            print(f"  predicted: error={pred.error} (load={t_load:.1f}s)")
            summary.append((name, "prediction_error", pred.error))
        else:
            print(f"  predicted age = {pred.age:.1f} (load={t_load:.1f}s, predict={t_pred*1000:.0f}ms)")
            summary.append((name, "ok", f"age={pred.age:.1f}"))

        # Free memory.
        try:
            m.teardown()
        except Exception:
            pass
        del m

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, status, detail in summary:
        flag = {"ok": "✓", "skipped": "·",
                "load_failed": "✗", "predict_failed": "✗",
                "prediction_error": "?"}.get(status, "?")
        print(f"  {flag} {name:<30s} {status:<18s} {detail}")
    n_ok = sum(1 for _, s, _ in summary if s == "ok")
    n_total = sum(1 for _, s, _ in summary if s != "skipped")
    print(f"\n{n_ok}/{n_total} models predicted successfully.")
    return 0 if n_ok == n_total else 2


if __name__ == "__main__":
    sys.exit(main())
