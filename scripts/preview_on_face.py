"""Preview every configured manipulation on a single face image.

Usage:
  python -m scripts.preview_on_face --image /path/to/face.jpg
  python -m scripts.preview_on_face --image /path/to/face.jpg --out outputs/preview.png

The output is a tiled grid: original + each manipulation, with labels.
This is the main visual sanity-check before running the full pipeline.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Importing classical registers all overlay-based manipulations.
import manipulations.classical  # noqa: F401
from manipulations.base import get_manipulation, list_manipulations
from manipulations.landmarks import detect_face_context
from utils import get_logger

LOG = get_logger(__name__)


def _label_strip(text: str, w: int, h: int = 24,
                 bg=(245, 245, 245), fg=(20, 20, 20)) -> np.ndarray:
    strip = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(strip, text, (6, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)
    return strip


def tile(images: list[tuple[str, np.ndarray]], cols: int = 4,
         pad: int = 4, bg=(255, 255, 255)) -> np.ndarray:
    """Lay out (label, image) pairs in a grid."""
    if not images:
        raise ValueError("Nothing to tile.")
    h, w = images[0][1].shape[:2]
    label_h = 24
    rows = math.ceil(len(images) / cols)
    canvas_h = rows * (h + label_h + pad) + pad
    canvas_w = cols * (w + pad) + pad
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    for i, (label, img) in enumerate(images):
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + label_h + pad)
        canvas[y:y + label_h, x:x + w] = _label_strip(label, w, label_h)
        canvas[y + label_h:y + label_h + h, x:x + w] = img
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply every classical manipulation to one image and tile the results.")
    parser.add_argument("--image", required=True, help="Path to an input face image.")
    parser.add_argument("--out", default=None,
                        help="Output path for the tiled preview "
                             "(default: outputs/preview_<input-stem>.png).")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--manipulations", nargs="*", default=None,
                        help="Subset of manipulations to apply (default: all classical).")
    args = parser.parse_args(argv)

    in_path = Path(args.image)
    if not in_path.exists():
        LOG.error("Image not found: %s", in_path)
        return 1
    img = cv2.imread(str(in_path))
    if img is None:
        LOG.error("Could not read %s", in_path)
        return 1

    LOG.info("Detecting landmarks…")
    ctx = detect_face_context(img)
    if ctx.landmarks is None:
        LOG.error("No face / landmarks detected. Try a clearer / larger image.")
        return 1
    LOG.info("Found face: bbox=%s, %d landmarks", ctx.bbox, len(ctx.landmarks))

    manip_names = args.manipulations or list_manipulations()
    LOG.info("Applying %d manipulations: %s", len(manip_names), manip_names)

    tiles: list[tuple[str, np.ndarray]] = [("original", img)]
    for name in manip_names:
        try:
            m = get_manipulation(name)
            out = m.apply(img, ctx)
            tiles.append((name, out))
        except Exception as e:
            LOG.exception("Manipulation %s failed: %s", name, e)

    grid = tile(tiles, cols=args.cols)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "outputs" / f"preview_{in_path.stem}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
