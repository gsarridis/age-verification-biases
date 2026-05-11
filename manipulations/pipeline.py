"""Apply configured manipulations to every image in Set B.

For each input image:
  1. Detect landmarks (once).
  2. Save the original (so the original and manipulated versions sit in the same dir).
  3. Apply each configured manipulation and save the result.

Outputs a manifest that maps (sample_id, manipulation) -> output path.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# Importing classical & genai modules registers their manipulations.
import manipulations.classical  # noqa: F401
from manipulations.base import get_manipulation, list_manipulations
from manipulations.landmarks import detect_face_context
from utils import Paths, ensure_dir, get_logger

LOG = get_logger(__name__)

# We import genai lazily, only if it's enabled (it pulls in torch + diffusers).


def _maybe_register_genai(cfg: dict) -> None:
    if cfg["manipulations"].get("genai", {}).get("enabled", False):
        from manipulations import genai  # noqa: F401
        from manipulations.genai import configure_sd
        g = cfg["manipulations"]["genai"]
        configure_sd(
            model_id=g.get("model_id"),
            guidance_scale=g.get("guidance_scale"),
            num_inference_steps=g.get("num_inference_steps"),
            strength=g.get("strength"),
        )


def _gather_manipulation_names(cfg: dict) -> list[str]:
    names: list[str] = []
    if cfg["manipulations"].get("classical", {}).get("enabled", False):
        names.extend(cfg["manipulations"]["classical"]["list"])
    if cfg["manipulations"].get("genai", {}).get("enabled", False):
        names.extend(cfg["manipulations"]["genai"]["list"])
    # De-dup while preserving order.
    seen: set[str] = set()
    out = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)

    available = list_manipulations()
    missing = [n for n in out if n not in available]
    if missing:
        raise KeyError(f"Configured manipulation(s) not registered: {missing}. "
                       f"Available: {available}")
    return out


def apply_to_manifest(manifest_csv: Path, cfg: dict) -> Path:
    """Apply manipulations to every row of a Set B manifest. Returns output manifest path."""
    paths = Paths.from_config(cfg)
    _maybe_register_genai(cfg)
    manip_names = _gather_manipulation_names(cfg)
    manips = {n: get_manipulation(n) for n in manip_names}
    LOG.info("Applying %d manipulations: %s", len(manips), list(manips))

    df = pd.read_csv(manifest_csv)
    out_dir = ensure_dir(paths.manipulated_dir)

    rows: list[dict] = []
    n_no_face = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Manipulating"):
        sample_id = row["sample_id"]
        src = Path(row["path"])
        img = cv2.imread(str(src))
        if img is None:
            LOG.warning("Could not read %s", src)
            continue

        ctx = detect_face_context(img)
        if ctx.landmarks is None:
            n_no_face += 1
            # We still record the original so it can serve as a baseline.
            sample_dir = ensure_dir(out_dir / sample_id)
            orig_out = sample_dir / "original.jpg"
            cv2.imwrite(str(orig_out), img)
            rows.append({**row.to_dict(), "manipulation": "original",
                         "manipulated_path": str(orig_out), "had_landmarks": False})
            continue

        sample_dir = ensure_dir(out_dir / sample_id)
        # Save original.
        orig_out = sample_dir / "original.jpg"
        cv2.imwrite(str(orig_out), img)
        rows.append({**row.to_dict(), "manipulation": "original",
                     "manipulated_path": str(orig_out), "had_landmarks": True})

        # Apply each manipulation.
        for mname, mobj in manips.items():
            try:
                out_img = mobj.apply(img, ctx)
            except Exception as e:
                LOG.exception("Manipulation %s failed on %s: %s", mname, sample_id, e)
                continue
            out_path = sample_dir / f"{mname}.jpg"
            cv2.imwrite(str(out_path), out_img)
            rows.append({**row.to_dict(), "manipulation": mname,
                         "manipulated_path": str(out_path), "had_landmarks": True})

    out_manifest = paths.manifests_dir / "set_b_manipulated.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_manifest, index=False)
    LOG.info("Wrote manipulated manifest: %d rows -> %s", len(out_df), out_manifest)
    if n_no_face:
        LOG.warning("%d/%d Set B images had no detected face — manipulations skipped for those.",
                    n_no_face, len(df))
    return out_manifest
