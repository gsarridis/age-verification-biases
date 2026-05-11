"""Generate a demo showcase figure using pre-rendered tiles + mock predictions.

This is NOT part of the experimental pipeline — it just produces a representative
example of what the showcase figure looks like, so users can see the output format
before running the real models.

It loads the pre-rendered preview_astronaut.jpg (which was generated earlier with all
manipulations correctly applied) and crops the tiles for original, mustache_thin,
beard_stubble, and eye_makeup_heavy. It then overlays mock predictions so the
red/green color coding at the 13-year threshold is visible.

Output: outputs/showcase_demo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THRESHOLD = 13
SOURCE_PREVIEW = ROOT / "outputs" / "preview_astronaut.jpg"

# Layout of the source preview grid (matches scripts/preview_on_face.py output).
# 4 columns × N rows; each cell = 24px label strip + 512x512 image + 4px padding.
GRID_COLS = 4
PAD = 4
LABEL_H = 24
TILE_H = 512
TILE_W = 512

# (manipulation_name, row_index, col_index in the source preview grid)
TILES = {
    "original":         (0, 0),
    "beard_stubble":    (0, 2),
    "eye_makeup_heavy": (0, 3),
    "mustache_thin":    (2, 0),
}


# Hand-tuned mock predictions illustrating the failure modes this study surfaces.
# True age = 9 (hypothetical).
MOCK_TRUE_AGE = 9
MOCK_PREDS = {
    # manipulation -> {model: predicted_age}
    "original":         {"fairface": 10.4, "mivolo": 9.1,  "vit": 11.5},
    "mustache_thin":    {"fairface": 14.2, "mivolo": 14.8, "vit": 16.8},
    "beard_stubble":    {"fairface": 12.3, "mivolo": 13.1, "vit": 15.0},
    "eye_makeup_heavy": {"fairface": 11.1, "mivolo": 10.4, "vit": 13.6},
}
MODEL_ORDER = ["fairface", "mivolo", "vit"]
MANIP_ORDER = ["original", "mustache_thin", "beard_stubble", "eye_makeup_heavy"]


def _crop_tile(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract a single tile from the 4-col preview grid."""
    cell_h = LABEL_H + TILE_H + PAD
    cell_w = TILE_W + PAD
    y0 = PAD + row * cell_h + LABEL_H
    x0 = PAD + col * cell_w
    return grid[y0:y0 + TILE_H, x0:x0 + TILE_W].copy()


def _format_pred(pred: float) -> tuple[str, str]:
    if np.isnan(pred):
        return ("—", "gray")
    color = "#cc1f1f" if pred >= THRESHOLD else "#1f7a1f"
    return f"{pred:.1f}", color


def main() -> int:
    if not SOURCE_PREVIEW.exists():
        print(f"ERROR: source preview not found: {SOURCE_PREVIEW}\n"
              "Run `python -m scripts.preview_on_face --image <face.jpg>` first.")
        return 1

    grid = cv2.imread(str(SOURCE_PREVIEW))
    if grid is None:
        print(f"ERROR: could not read {SOURCE_PREVIEW}")
        return 1

    # Extract tiles for the four manipulations we care about.
    images: dict[str, np.ndarray] = {}
    for manip in MANIP_ORDER:
        if manip not in TILES:
            continue
        row, col = TILES[manip]
        images[manip] = _crop_tile(grid, row, col)

    # Build the figure: 1 row × 4 columns, with predictions stacked under each tile.
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6),
                             gridspec_kw={"wspace": 0.18, "top": 0.86, "bottom": 0.05})

    fig.suptitle(
        "DEMO: Age verification under simple manipulations\n"
        f"subject_demo (true age = {MOCK_TRUE_AGE})    "
        f"red = predicted ≥ {THRESHOLD} (would pass adult check)   "
        f"green = predicted < {THRESHOLD}",
        fontsize=11, y=0.97,
    )

    for j, manip in enumerate(MANIP_ORDER):
        ax = axes[j]
        if manip not in images:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_axis_off()
            continue
        rgb = cv2.cvtColor(images[manip], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(manip.replace("_", " "), fontsize=10, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        for k, model in enumerate(MODEL_ORDER):
            pred = MOCK_PREDS[manip].get(model, float("nan"))
            text, color = _format_pred(pred)
            ax.text(0.0, -0.06 - 0.08 * k,
                    f"{model}: {text}",
                    transform=ax.transAxes, fontsize=11,
                    color=color, ha="left", va="top",
                    family="monospace")

    out_path = ROOT / "outputs" / "showcase_demo.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote demo showcase: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
