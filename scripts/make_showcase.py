"""Generate the 'showcase' figure for the post.

Selects a small number of representative samples from Set B, and for each one builds a
panel showing:
  * original image + each manipulation,
  * each model's predicted age,
  * the ground-truth age,
  * a clear visual marker for predictions that crossed the 13-year threshold.

Usage:
    python -m scripts.make_showcase --config configs/default.yaml [--n 5]

Selection strategy (heuristic; tuned to produce *interesting* examples for the post):
  1. Compute, for every Set B sample, the maximum Δ predicted age across all
     (model, manipulation) pairs — i.e., how badly the manipulations confuse models on
     this particular subject.
  2. Take the top-N by that score (with a diversity constraint: spread true ages
     across the 6-12 range).

The figure is saved to ``<report_dir>/showcase.png`` and as a horizontally-tiled set of
per-sample PNGs to ``<report_dir>/showcase/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._args import add_common_args
from utils import Paths, apply_overrides, ensure_dir, get_logger, load_config

LOG = get_logger(__name__)

THRESHOLD = 13


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def _load_predictions(
    paths: Paths, manipulation_filter: list[str] | None = None
) -> pd.DataFrame:
    """Concatenate all set_b_manipulated__*.csv prediction files into one long DataFrame.

    If ``manipulation_filter`` is provided, only those manipulations (plus 'original')
    are kept.
    """
    rows = []
    for p in sorted(paths.predictions_dir.glob("set_b_manipulated__*.csv")):
        model = p.stem.split("__", 1)[1]
        df = pd.read_csv(p)
        df["model"] = model
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if manipulation_filter:
        keep = set(manipulation_filter) | {"original"}
        out = out[out["manipulation"].isin(keep)].reset_index(drop=True)
    return out


def _select_showcase_samples(
    preds: pd.DataFrame, n: int = 10, age_diversity: bool = True
) -> list[str]:
    """Pick N sample_ids that best illustrate manipulation-induced age inflation.

    Score each sample by the max Δ predicted age across (model × manipulation),
    where Δ = pred_manipulated − pred_original. Higher score = more dramatic flip.
    """
    if preds.empty:
        return []

    df = preds.dropna(subset=["predicted_age"])
    originals = df[df["manipulation"] == "original"][
        ["sample_id", "model", "predicted_age", "true_age"]
    ].rename(columns={"predicted_age": "pred_original"})
    others = df[df["manipulation"] != "original"]
    merged = others.merge(
        originals[["sample_id", "model", "pred_original"]],
        on=["sample_id", "model"],
        how="inner",
    )
    merged["delta"] = merged["predicted_age"] - merged["pred_original"]

    # Score = mean delta per sample, but also weight strongly toward samples where at
    # least one (model, manip) actually crossed the 13 threshold.
    merged["crossed"] = (
        (merged["pred_original"] < THRESHOLD) & (merged["predicted_age"] >= THRESHOLD)
    ).astype(int)
    per_sample = (
        merged.groupby("sample_id")
        .agg(
            max_delta=("delta", "max"),
            mean_delta=("delta", "mean"),
            any_crossed=("crossed", "max"),
            n_crossed=("crossed", "sum"),
            true_age=("true_age", "first"),
        )
        .reset_index()
    )

    # Sort by (any_crossed desc, n_crossed desc, max_delta desc).
    per_sample = per_sample.sort_values(
        ["any_crossed", "n_crossed", "max_delta"],
        ascending=[False, False, False],
    )

    if not age_diversity:
        return per_sample["sample_id"].head(n).tolist()

    # Try to spread across true ages (e.g., 6, 8, 10, 12, ...).
    selected: list[str] = []
    used_ages: set[int] = set()
    # First pass: prefer samples whose age is not yet represented.
    for _, row in per_sample.iterrows():
        if len(selected) >= n:
            break
        age = int(row["true_age"])
        if age not in used_ages:
            selected.append(row["sample_id"])
            used_ages.add(age)
    # Second pass: fill remaining slots with the highest-scoring samples regardless.
    if len(selected) < n:
        for _, row in per_sample.iterrows():
            if len(selected) >= n:
                break
            if row["sample_id"] not in selected:
                selected.append(row["sample_id"])
    selected = selected[:n]
    selected = [selected[i] for i in [2, 4, 8]]
    return selected


# ---------------------------------------------------------------------------
# Image loading (from manipulated manifest)
# ---------------------------------------------------------------------------


def _load_sample_images(
    sample_id: str, manifest: pd.DataFrame, manipulation_filter: list[str] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ``({manipulation_name: rgb_image}, {manipulation_name: status_string})``.

    The status dict records why a manipulation may be missing, so the figure can show a
    useful label instead of a generic "missing" placeholder. Possible status values:
      * "ok"            — image loaded successfully
      * "not_in_manifest" — manipulation has predictions but no manifest row
      * "file_not_found"  — manifest row exists but cv2.imread returned None
    """
    rows = manifest[manifest["sample_id"] == sample_id]
    out: dict[str, np.ndarray] = {}
    status: dict[str, str] = {}
    for _, r in rows.iterrows():
        manip = r["manipulation"]
        path = r["manipulated_path"]
        keep = set(manipulation_filter) | {"original"}
        # print(keep)
        for m in list(keep):
            manip = m
            path = r["manipulated_path"]
            path = path.replace("original", m)
            # print(path)
            if not Path(path).exists():
                status[manip] = "file_not_found"
                continue
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                status[manip] = "file_not_found"
                continue
            out[manip] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            status[manip] = "ok"
    return out, status


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _format_pred(pred: float, threshold: int = THRESHOLD) -> tuple[str, str]:
    """Return (text, color) for a prediction. Red if >= threshold (misclassified)."""
    if np.isnan(pred):
        return ("—", "gray")
    text = f"{pred:.1f}"
    color = "#cc1f1f" if pred >= threshold else "#1f7a1f"
    return text, color


def _draw_sample_panel(
    ax_row,
    sample_id: str,
    true_age: int,
    images: dict[str, np.ndarray],
    status: dict[str, str],
    preds_by_manip: dict[str, dict[str, float]],
    manip_order: list[str],
    model_order: list[str],
) -> None:
    """Render one row: original + each manipulation, with predictions printed below each tile."""
    cols = ["original"] + manip_order
    for j, manip in enumerate(cols):
        ax = ax_row[j]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.set_title(manip.replace("_", " "), fontsize=10, pad=4)

        if manip not in images:
            why = status.get(manip)
            label = {
                "file_not_found": "image file\nnot on disk",
                "not_in_manifest": "manipulation\nnot applied",
            }.get(why, "missing")
            ax.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
                fontsize=10,
            )
            # Hide the empty frame for the missing tile.
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        ax.imshow(images[manip])

        # Predictions per model, printed BELOW the image (axes y > 1 is above; y < 0 is below
        # when origin is top-left after imshow — matplotlib's default origin is 'upper' for
        # imshow, but ax.transAxes is always (0,0)=bottom-left, (1,1)=top-left. We want
        # below the image, so we use negative y in axes coords.)
        for k, model in enumerate(model_order):
            pred = preds_by_manip.get(manip, {}).get(model, float("nan"))
            text, color = _format_pred(pred)
            ax.text(
                0.0,
                -0.04 - 0.08 * k,
                f"{model}: {text}",
                transform=ax.transAxes,
                fontsize=9,
                color=color,
                ha="left",
                va="top",
                family="monospace",
            )

    # Row label (left side): sample_id + true age.
    ax_row[0].set_ylabel(
        f"true age = {true_age}",
        rotation=0,
        ha="right",
        va="center",
        labelpad=42,
        fontsize=10,
    )


def make_showcase(
    cfg: dict,
    n_samples: int = 5,
    manip_order: list[str] | None = None,
    out_path: Path | None = None,
) -> Path:
    paths = Paths.from_config(cfg)

    # The same filter used by evaluate_all, so the showcase reflects the same scope
    # as the metrics tables in the report.
    manip_filter = cfg.get("evaluation", {}).get("manipulation_filter") or None

    preds = _load_predictions(paths, manipulation_filter=manip_filter)
    if preds.empty:
        raise RuntimeError(
            "No Set B predictions found at "
            f"{paths.predictions_dir}. Run scripts.run_models first."
        )

    # Build the manifest of where each manipulated image lives on disk.
    set_b_man_csv = paths.manifests_dir / "set_b_manipulated.csv"
    if not set_b_man_csv.exists():
        raise RuntimeError(f"No manipulated manifest at {set_b_man_csv}.")
    manifest = pd.read_csv(set_b_man_csv)

    # Manipulation order in the figure: prefer the explicit filter (config or arg),
    # then the classical list, then whatever's in the predictions.
    if manip_order is None:
        if manip_filter:
            manip_order = list(manip_filter)
        else:
            manip_order = list(cfg["manipulations"]["classical"]["list"])
    # Drop manipulations that have no predictions (e.g., misspelled in config).
    available = set(preds["manipulation"].unique()) - {"original"}
    manip_order = [m for m in manip_order if m in available]
    if not manip_order:
        raise RuntimeError(
            "After filtering, no manipulations remain to display. "
            f"Available in predictions: {sorted(available)}; filter: {manip_filter}"
        )

    # Models = whichever models actually have predictions on Set B.
    model_order = sorted(preds["model"].unique())

    # Pick samples.
    sample_ids = _select_showcase_samples(preds, n=n_samples)
    LOG.info("Showcase samples: %s", sample_ids)

    if not sample_ids:
        raise RuntimeError("Could not select any showcase samples.")

    # Build the figure: rows = samples, cols = (original + manipulations).
    n_cols = 1 + len(manip_order)
    fig_w = 3.2 * n_cols
    # Row height: image (~3 in) + 3 prediction lines (~0.25 in each) + title (~0.3 in).
    fig_h = 3.9 * len(sample_ids)
    fig, axes = plt.subplots(
        len(sample_ids),
        n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "hspace": 0.55,
            "wspace": 0.20,
            "left": 0.10,
            "right": 0.99,
            "top": 0.94,
            "bottom": 0.03,
        },
        squeeze=False,
    )

    # Header row (above the first row of images): "ground truth" legend.
    # fig.suptitle(
    #     "Age verification under simple manipulations\n"
    #     f"red = predicted ≥ {THRESHOLD} (would pass adult check)   "
    #     f"green = predicted < {THRESHOLD}",
    #     fontsize=12,
    #     y=0.99,
    # )

    # Diagnostic: count how often each manipulation actually loaded vs was missing.
    diag_loaded: dict[str, int] = {m: 0 for m in manip_order}
    diag_not_in_manifest: dict[str, int] = {m: 0 for m in manip_order}
    diag_file_not_found: dict[str, int] = {m: 0 for m in manip_order}

    for i, sid in enumerate(sample_ids):
        sub = preds[preds["sample_id"] == sid]
        true_age = int(sub["true_age"].iloc[0])

        # preds_by_manip[manip][model] = pred
        preds_by_manip: dict[str, dict[str, float]] = {}
        for _, r in sub.iterrows():
            preds_by_manip.setdefault(r["manipulation"], {})[r["model"]] = r[
                "predicted_age"
            ]

        images, status = _load_sample_images(sid, manifest, manip_filter)
        # For every manipulation we *expected* to display but didn't see in the manifest
        # at all, mark it as not_in_manifest so the panel renders a clearer label.
        for m in manip_order + ["original"]:
            if m not in status:
                status[m] = "not_in_manifest"
        # Update the per-manipulation diagnostic counters.
        for m in manip_order:
            s = status.get(m)
            if s == "ok":
                diag_loaded[m] += 1
            elif s == "not_in_manifest":
                diag_not_in_manifest[m] += 1
            elif s == "file_not_found":
                diag_file_not_found[m] += 1

        _draw_sample_panel(
            axes[i],
            sid,
            true_age,
            images,
            status,
            preds_by_manip,
            manip_order,
            model_order,
        )

    # Log the diagnostic summary — helps users see immediately whether "missing"
    # tiles mean the manipulation was never applied or the file is gone.
    LOG.info("Tile load summary across %d sample(s):", len(sample_ids))
    for m in manip_order:
        LOG.info(
            "  %-22s loaded=%d  not_in_manifest=%d  file_not_found=%d",
            m,
            diag_loaded[m],
            diag_not_in_manifest[m],
            diag_file_not_found[m],
        )

    # Save.
    if out_path is None:
        out_path = paths.report_dir / "showcase.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote showcase: %s", out_path)
    return out_path


def diagnose(cfg: dict, n_samples: int = 5) -> int:
    """Print exactly what the script can see, without rendering anything.

    Useful when the figure shows 'missing' tiles unexpectedly: this tells you whether
    the issue is missing manifest rows, missing image files on disk, or something else.
    """
    paths = Paths.from_config(cfg)
    manip_filter = cfg.get("evaluation", {}).get("manipulation_filter") or None

    preds = _load_predictions(paths, manipulation_filter=manip_filter)
    set_b_man_csv = paths.manifests_dir / "set_b_manipulated.csv"

    print(f"\n=== Predictions ===")
    print(f"  predictions_dir: {paths.predictions_dir}")
    if preds.empty:
        print("  (no Set B predictions found)")
        return 1
    print(f"  rows: {len(preds)}")
    print(f"  models: {sorted(preds['model'].unique())}")
    print(f"  manipulations in predictions: {sorted(preds['manipulation'].unique())}")
    if manip_filter:
        print(f"  filter (from config): {manip_filter}")

    print(f"\n=== Manifest ===")
    print(f"  path: {set_b_man_csv}  (exists: {set_b_man_csv.exists()})")
    if not set_b_man_csv.exists():
        print(
            "  CANNOT FIND THE MANIPULATED MANIFEST. Showcase needs this to locate "
            "the image files on disk."
        )
        return 2
    manifest = pd.read_csv(set_b_man_csv)
    print(f"  rows: {len(manifest)}")
    print(f"  manipulations in manifest: {sorted(manifest['manipulation'].unique())}")

    in_preds = set(preds["manipulation"].unique())
    in_manifest = set(manifest["manipulation"].unique())
    only_preds = in_preds - in_manifest
    only_manifest = in_manifest - in_preds
    if only_preds:
        print(
            f"\n  ! manipulations in PREDICTIONS but not in MANIFEST: {sorted(only_preds)}"
        )
        print("    -> the showcase cannot find image files for these.")
        print("    -> re-run scripts.apply_manipulations to regenerate the manifest,")
        print("       or update set_b_manipulated.csv to include these manipulations.")
    if only_manifest:
        print(
            f"\n  manipulations in MANIFEST but not in PREDICTIONS: {sorted(only_manifest)}"
        )

    # Sample-level disk check.
    sample_ids = _select_showcase_samples(preds, n=n_samples)
    print(f"\n=== Sample disk check (top {len(sample_ids)} by Δ score) ===")
    for sid in sample_ids:
        rows = manifest[manifest["sample_id"] == sid]
        print(f"  {sid}: {len(rows)} manifest row(s)")
        for _, r in rows.iterrows():
            p = Path(r["manipulated_path"])
            tag = "OK" if p.exists() else "MISSING"
            print(f"    [{tag}] {r['manipulation']:<22s} {r['manipulated_path']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the showcase figure.")
    add_common_args(p)
    p.add_argument(
        "--n", type=int, default=10, help="Number of example samples to show."
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Print what the script sees on disk and exit (don't render).",
    )
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    if args.diagnose:
        return diagnose(cfg, n_samples=args.n)
    make_showcase(cfg, n_samples=args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
