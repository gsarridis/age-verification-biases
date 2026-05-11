"""Visualize the procedural overlays on transparent canvases.

This script does NOT require any face dataset — it just dumps each overlay
stamp as a PNG so you can see what the classical manipulations look like
before they are blended onto a face.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manipulations.classical import (
    _make_mustache_stamp,
    _make_glasses_stamp,
    _make_hat_stamp,
)


def composite_on_checkerboard(stamp, cell=10):
    """Place an alpha-stamp on a checkerboard background (so transparency is visible)."""
    h, w = stamp.shape[:2]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            shade = 230 if ((x // cell) + (y // cell)) % 2 == 0 else 200
            bg[y:y + cell, x:x + cell] = shade
    alpha = (stamp[..., 3:4].astype(np.float32) / 255.0)
    return (bg.astype(np.float32) * (1 - alpha) +
            stamp[..., :3].astype(np.float32) * alpha).astype(np.uint8)


def main():
    out = Path(__file__).resolve().parent.parent / "outputs" / "overlay_previews"
    out.mkdir(parents=True, exist_ok=True)

    items = [
        ("mustache_thin", _make_mustache_stamp(120, thickness=0.6)),
        ("mustache_thick", _make_mustache_stamp(120, thickness=1.4)),
        ("glasses", _make_glasses_stamp(180)),
        ("hat", _make_hat_stamp(220)),
    ]
    for name, stamp in items:
        png_path = out / f"{name}.png"
        cv2.imwrite(str(png_path), stamp)             # Saves with alpha.
        preview = composite_on_checkerboard(stamp)
        cv2.imwrite(str(out / f"{name}_preview.png"), preview)
        print(f"  wrote {png_path} (shape={stamp.shape})")

    print(f"\nAll overlays written to {out}")


if __name__ == "__main__":
    main()
