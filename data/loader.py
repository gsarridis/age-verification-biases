"""UTKFace dataset loader.

UTKFace filenames follow the pattern ``[age]_[gender]_[race]_[datetime].jpg(.chip.jpg)``,
e.g. ``7_0_3_20170109150557335.jpg.chip.jpg``.

Some files in the original release are malformed (missing fields). We skip those and log
a count.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import get_logger

LOG = get_logger(__name__)

# age, gender, race, then anything (datetime + extension variants).
UTKFACE_NAME_RE = re.compile(r"^(\d{1,3})_(\d)_(\d)_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

GENDER_MAP = {0: "male", 1: "female"}
RACE_MAP = {
    0: "white",
    1: "black",
    2: "asian",
    3: "indian",
    4: "other",
}


def _iter_image_files(root: Path) -> Iterable[Path]:
    # UTKFace ships as a flat folder, but we glob recursively so the loader still works
    # if users have organized into subfolders.
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        yield from root.rglob(ext)


def load_utkface(root: str | Path) -> pd.DataFrame:
    """Scan a UTKFace directory and return a DataFrame.

    Columns: ``path, age, gender, race, filename``.
    Rows whose filename does not match the UTKFace pattern are skipped (with a warning).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"UTKFace root does not exist: {root}")

    rows: list[dict] = []
    skipped = 0
    for p in _iter_image_files(root):
        m = UTKFACE_NAME_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        age, gender, race = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Sanity bounds. UTKFace claims 0-116 but a few files have annotation noise.
        if age < 0 or age > 120:
            skipped += 1
            continue
        rows.append({
            "path": str(p.resolve()),
            "filename": p.name,
            "age": age,
            "gender": GENDER_MAP.get(gender, "unknown"),
            "race": RACE_MAP.get(race, "unknown"),
        })

    if not rows:
        raise RuntimeError(
            f"No UTKFace-formatted images found under {root}. "
            "Check that the path is correct and that filenames follow the "
            "UTKFace [age]_[gender]_[race]_*.jpg convention."
        )

    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    LOG.info("Loaded UTKFace: %d images (%d skipped) from %s",
             len(df), skipped, root)
    LOG.info("Age range: %d-%d (mean=%.1f)",
             df["age"].min(), df["age"].max(), df["age"].mean())
    LOG.info("Children (<13): %d (%.1f%%)",
             (df["age"] < 13).sum(), 100 * (df["age"] < 13).mean())
    return df


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Top-level dispatcher. Add other datasets here as needed."""
    name = cfg["dataset"]["name"].lower()
    if name == "utkface":
        return load_utkface(cfg["dataset"]["root"])
    raise NotImplementedError(f"Dataset '{name}' not implemented.")
