"""Metrics for the age-verification bias study.

This module answers three concrete questions:

  1. **Set A (MAE)** — How accurate is each model overall on a balanced minor/adult set?
     Reported overall + split into minors / adults so you can see if the error is
     systematically larger for one group.

  2. **Set B (Δ predicted age)** — Per (model × manipulation), how much does the model's
     predicted age shift relative to its prediction on the *original* image of the same
     subject? Reported as mean Δ years and median Δ years, with a 95% CI on the mean.

  3. **Set B (binary @ 13)** — Treat age verification as a binary problem: does the
     model classify the subject as <13 or ≥13? Compute accuracy / per-class confusion
     on (a) the original images and (b) each manipulation. The drop in accuracy from
     original → manipulated is the headline number for the post.

All three tables are written to ``<output_dir>/metrics/`` as CSVs and consumed by the
report generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils import Paths, get_logger

LOG = get_logger(__name__)

THRESHOLD = 13  # Facebook's minimum age — the headline binary boundary.


# ===========================================================================
# Low-level helpers
# ===========================================================================


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_iters: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_iters, n))
    means = np.nanmean(values[idx], axis=1)
    lo, hi = np.nanquantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_proportion_ci(
    successes: np.ndarray,
    n_iters: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a proportion (successes is a boolean / {0,1} array)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(successes)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_iters, n))
    props = np.mean(successes[idx], axis=1)
    lo, hi = np.nanquantile(props, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ===========================================================================
# Question 1: Set A overall MAE
# ===========================================================================


@dataclass
class SetAMetrics:
    model: str
    n: int
    n_minors: int
    n_adults: int
    mae_overall: float
    mae_overall_lo: float
    mae_overall_hi: float
    mae_minors: float
    mae_minors_lo: float
    mae_minors_hi: float
    mae_adults: float
    mae_adults_lo: float
    mae_adults_hi: float
    mean_predicted_age_minors: float
    mean_predicted_age_adults: float


def evaluate_set_a(
    pred_csv: Path,
    model_name: str,
    bootstrap_iters: int = 1000,
    threshold: int = THRESHOLD,
) -> SetAMetrics:
    df = pd.read_csv(pred_csv).dropna(subset=["predicted_age"])
    y_true = df["true_age"].to_numpy(dtype=float)
    y_pred = df["predicted_age"].to_numpy(dtype=float)

    minor_mask = y_true < threshold
    adult_mask = ~minor_mask
    abs_err = np.abs(y_true - y_pred)

    rng = np.random.default_rng(0)
    mae_o_lo, mae_o_hi = bootstrap_mean_ci(abs_err, n_iters=bootstrap_iters, rng=rng)
    mae_m_lo, mae_m_hi = bootstrap_mean_ci(
        abs_err[minor_mask], n_iters=bootstrap_iters, rng=rng
    )
    mae_a_lo, mae_a_hi = bootstrap_mean_ci(
        abs_err[adult_mask], n_iters=bootstrap_iters, rng=rng
    )

    return SetAMetrics(
        model=model_name,
        n=len(df),
        n_minors=int(minor_mask.sum()),
        n_adults=int(adult_mask.sum()),
        mae_overall=mae(y_true, y_pred),
        mae_overall_lo=mae_o_lo,
        mae_overall_hi=mae_o_hi,
        mae_minors=mae(y_true[minor_mask], y_pred[minor_mask]),
        mae_minors_lo=mae_m_lo,
        mae_minors_hi=mae_m_hi,
        mae_adults=mae(y_true[adult_mask], y_pred[adult_mask]),
        mae_adults_lo=mae_a_lo,
        mae_adults_hi=mae_a_hi,
        mean_predicted_age_minors=(
            float(np.mean(y_pred[minor_mask])) if minor_mask.any() else float("nan")
        ),
        mean_predicted_age_adults=(
            float(np.mean(y_pred[adult_mask])) if adult_mask.any() else float("nan")
        ),
    )


# ===========================================================================
# Question 2: Set B — how much does each manipulation shift predicted age?
# ===========================================================================
#
# For each subject in Set B, we have its predicted age on the original image and its
# predicted age on each manipulation. The Δ for one (subject, manipulation, model)
# is delta = pred_manipulated - pred_original. We summarize across subjects.


@dataclass
class DeltaPredictionMetrics:
    model: str
    manipulation: str
    n_paired: int  # Number of subjects with both original & manipulation.
    # --- Δ relative to model's own prediction on the original image ---
    # "How much does the manipulation shift the model's answer?"
    mean_delta_years: float
    mean_delta_years_lo: float
    mean_delta_years_hi: float
    median_delta_years: float
    pct_subjects_aged_up: float  # Fraction of subjects whose predicted age went UP.
    # --- Δ relative to ground-truth age ---
    # "How far from reality is the manipulated prediction?"
    # signed_error_original = pred_original - true_age   (mean over subjects)
    # signed_error_manipulated = pred_manipulated - true_age
    # delta_signed_error = signed_error_manipulated - signed_error_original
    #                    = (pred_manip - true) - (pred_orig - true)
    #                    = pred_manip - pred_orig
    # so by construction this is mathematically equal to mean_delta_years above; we report
    # the *raw* signed errors (vs. truth) as separate columns so readers can see both
    # baselines side-by-side without doing arithmetic.
    mean_signed_error_original: float  # mean(pred_orig - true_age)
    mean_signed_error_manipulated: float  # mean(pred_manip - true_age)
    mean_abs_error_original: float  # MAE on originals, for comparison.
    mean_abs_error_manipulated: float  # MAE on manipulated.
    # Convenience columns:
    mean_pred_original: float
    mean_pred_manipulated: float
    mean_true_age: float


def _build_paired_df(
    set_b_csv: Path, manipulation_filter: Optional[list[str]] = None
) -> pd.DataFrame:
    """For each (sample_id, manipulation), produce a row with original_pred &
    manipulated_pred side-by-side. Drops samples missing one or the other.

    ``manipulation_filter``: if provided, only manipulations in this list (plus the
    implicit "original" baseline) are considered. Use this to scope down a prediction
    CSV that contains more manipulations than the current report should show.
    """
    df = pd.read_csv(set_b_csv).dropna(subset=["predicted_age"])
    if manipulation_filter:
        keep = set(manipulation_filter) | {"original"}
        df = df[df["manipulation"].isin(keep)]
    originals = df[df["manipulation"] == "original"][
        ["sample_id", "true_age", "predicted_age"]
    ].rename(columns={"predicted_age": "pred_original"})
    others = df[df["manipulation"] != "original"].rename(
        columns={"predicted_age": "pred_manipulated"}
    )
    merged = others.merge(
        originals[["sample_id", "pred_original"]], on="sample_id", how="inner"
    )
    merged["delta"] = merged["pred_manipulated"] - merged["pred_original"]
    return merged


def evaluate_delta_predictions(
    pred_csv: Path,
    model_name: str,
    bootstrap_iters: int = 1000,
    manipulation_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Per-manipulation Δ-prediction metrics. Returns a DataFrame, one row per manipulation."""
    paired = _build_paired_df(pred_csv, manipulation_filter=manipulation_filter)
    if paired.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(0)
    rows = []
    for manip, sub in paired.groupby("manipulation"):
        deltas = sub["delta"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(deltas, n_iters=bootstrap_iters, rng=rng)

        signed_err_orig = (sub["pred_original"] - sub["true_age"]).to_numpy(dtype=float)
        signed_err_manip = (sub["pred_manipulated"] - sub["true_age"]).to_numpy(
            dtype=float
        )

        rows.append(
            DeltaPredictionMetrics(
                model=model_name,
                manipulation=manip,
                n_paired=len(sub),
                mean_delta_years=float(np.mean(deltas)),
                mean_delta_years_lo=lo,
                mean_delta_years_hi=hi,
                median_delta_years=float(np.median(deltas)),
                pct_subjects_aged_up=float(np.mean(deltas > 0)),
                mean_signed_error_original=float(np.mean(signed_err_orig)),
                mean_signed_error_manipulated=float(np.mean(signed_err_manip)),
                mean_abs_error_original=float(np.mean(np.abs(signed_err_orig))),
                mean_abs_error_manipulated=float(np.mean(np.abs(signed_err_manip))),
                mean_pred_original=float(np.mean(sub["pred_original"])),
                mean_pred_manipulated=float(np.mean(sub["pred_manipulated"])),
                mean_true_age=float(np.mean(sub["true_age"])),
            )
        )
    return pd.DataFrame([asdict(r) for r in rows])


# ===========================================================================
# Question 3: Set B — binary classification at 13 (Facebook threshold)
# ===========================================================================
#
# All Set B subjects have true age < 13. So the *correct* prediction is always "minor".
# The interesting numbers are:
#   * accuracy on originals    = fraction predicted < 13 on the unmanipulated image
#   * accuracy on manipulated  = fraction predicted < 13 on the manipulated image
#   * Δ accuracy = original − manipulated (>= 0; the bigger, the more the manipulation hurts)
#   * confusion: how many minors who were correctly classified on the original got flipped
#     to "adult" by the manipulation? This is the most concerning failure mode.


@dataclass
class BinaryThresholdMetrics:
    model: str
    manipulation: str
    threshold: int
    n_paired: int
    # Per-image accuracy: fraction predicted as MINOR (correct class for Set B).
    acc_original: float
    acc_original_lo: float
    acc_original_hi: float
    acc_manipulated: float
    acc_manipulated_lo: float
    acc_manipulated_hi: float
    delta_accuracy: (
        float  # acc_original - acc_manipulated (positive = manipulation hurt).
    )
    # Per-subject transitions (the "flip" analysis):
    n_correct_to_correct: int  # Predicted minor on both: robust correct cases.
    n_correct_to_wrong: (
        int  # Predicted minor on original, adult on manipulated: FLIPPED.
    )
    n_wrong_to_correct: (
        int  # Adult on original, minor on manipulated: model recovered (rare).
    )
    n_wrong_to_wrong: int  # Adult on both: persistent error.
    flip_rate: float  # n_correct_to_wrong / (n_correct_to_correct + n_correct_to_wrong)


def evaluate_binary_threshold(
    pred_csv: Path,
    model_name: str,
    threshold: int = THRESHOLD,
    bootstrap_iters: int = 1000,
    manipulation_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    paired = _build_paired_df(pred_csv, manipulation_filter=manipulation_filter)
    if paired.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(0)
    rows = []
    for manip, sub in paired.groupby("manipulation"):
        # All Set B subjects are minors, so "predicted minor" == "correct".
        correct_orig = (sub["pred_original"] < threshold).to_numpy()
        correct_manip = (sub["pred_manipulated"] < threshold).to_numpy()

        acc_o = float(np.mean(correct_orig))
        acc_m = float(np.mean(correct_manip))
        acc_o_lo, acc_o_hi = bootstrap_proportion_ci(
            correct_orig.astype(int), n_iters=bootstrap_iters, rng=rng
        )
        acc_m_lo, acc_m_hi = bootstrap_proportion_ci(
            correct_manip.astype(int), n_iters=bootstrap_iters, rng=rng
        )

        # Per-subject flip analysis.
        n_cc = int(np.sum(correct_orig & correct_manip))
        n_cw = int(np.sum(correct_orig & ~correct_manip))
        n_wc = int(np.sum(~correct_orig & correct_manip))
        n_ww = int(np.sum(~correct_orig & ~correct_manip))
        denom = n_cc + n_cw
        flip_rate = float(n_cw / denom) if denom > 0 else float("nan")

        rows.append(
            BinaryThresholdMetrics(
                model=model_name,
                manipulation=manip,
                threshold=threshold,
                n_paired=len(sub),
                acc_original=acc_o,
                acc_original_lo=acc_o_lo,
                acc_original_hi=acc_o_hi,
                acc_manipulated=acc_m,
                acc_manipulated_lo=acc_m_lo,
                acc_manipulated_hi=acc_m_hi,
                delta_accuracy=acc_o - acc_m,
                n_correct_to_correct=n_cc,
                n_correct_to_wrong=n_cw,
                n_wrong_to_correct=n_wc,
                n_wrong_to_wrong=n_ww,
                flip_rate=flip_rate,
            )
        )
    return pd.DataFrame([asdict(r) for r in rows])


# ===========================================================================
# Top-level driver: walks predictions/, writes metrics/
# ===========================================================================


def evaluate_all(cfg: dict) -> dict[str, Path]:
    paths = Paths.from_config(cfg)
    pred_dir = paths.predictions_dir
    metrics_dir = paths.metrics_dir
    boots = cfg["evaluation"].get("bootstrap_iters", 1000)
    thr = cfg["evaluation"].get("primary_threshold", THRESHOLD)
    manip_filter = cfg["evaluation"].get("manipulation_filter") or None
    if manip_filter:
        LOG.info("Set B evaluation restricted to manipulations: %s", manip_filter)

    out_paths: dict[str, Path] = {}

    # Set A: per-model MAE.
    set_a_rows = []
    for p in sorted(pred_dir.glob("set_a__*.csv")):
        model_name = p.stem.split("__", 1)[1]
        try:
            m = evaluate_set_a(p, model_name, bootstrap_iters=boots, threshold=thr)
            set_a_rows.append(asdict(m))
        except Exception as e:
            LOG.exception("Set A eval failed for %s: %s", p, e)
    if set_a_rows:
        out = metrics_dir / "set_a_mae.csv"
        pd.DataFrame(set_a_rows).to_csv(out, index=False)
        LOG.info("Wrote Set A MAE table: %s", out)
        out_paths["set_a_mae"] = out

    # Set B: Δ predictions and binary @ 13.
    delta_dfs, binary_dfs = [], []
    for p in sorted(pred_dir.glob("set_b_manipulated__*.csv")):
        model_name = p.stem.split("__", 1)[1]
        try:
            d = evaluate_delta_predictions(
                p, model_name, bootstrap_iters=boots, manipulation_filter=manip_filter
            )
            if not d.empty:
                delta_dfs.append(d)
        except Exception as e:
            LOG.exception("Δ-prediction eval failed for %s: %s", p, e)
        try:
            b = evaluate_binary_threshold(
                p,
                model_name,
                threshold=thr,
                bootstrap_iters=boots,
                manipulation_filter=manip_filter,
            )
            if not b.empty:
                binary_dfs.append(b)
        except Exception as e:
            LOG.exception("Binary-threshold eval failed for %s: %s", p, e)

    if delta_dfs:
        all_d = pd.concat(delta_dfs, ignore_index=True)
        out = metrics_dir / "set_b_delta_predictions.csv"
        all_d.to_csv(out, index=False)
        LOG.info("Wrote Set B Δ-prediction table: %s", out)
        out_paths["set_b_delta"] = out

    if binary_dfs:
        all_b = pd.concat(binary_dfs, ignore_index=True)
        out = metrics_dir / "set_b_binary_threshold.csv"
        all_b.to_csv(out, index=False)
        LOG.info("Wrote Set B binary @ %d table: %s", thr, out)
        out_paths["set_b_binary"] = out

    return out_paths
