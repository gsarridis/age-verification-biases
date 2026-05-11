"""Build the two test sets (A: balanced, B: minors-only) from a loaded dataset.

Both manifests are written as CSVs to ``<output_dir>/manifests/`` so that the rest of
the pipeline (manipulation, model evaluation, reporting) operates on stable file lists
and is fully reproducible given the seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import Paths, get_logger

LOG = get_logger(__name__)


def _stratified_sample(df: pd.DataFrame, n: int, by: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` rows, trying to spread evenly across the cross-product of ``by`` columns.

    Falls back to a uniform random sample if stratification yields fewer than n rows
    (e.g., very thin race × gender bins among children).
    """
    if not by or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=int(rng.integers(0, 2**31 - 1)))

    groups = df.groupby(by, dropna=False)
    n_groups = len(groups)
    base = n // n_groups
    remainder = n - base * n_groups

    sampled = []
    for _, g in groups:
        take = min(base, len(g))
        if take > 0:
            sampled.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    pulled = pd.concat(sampled) if sampled else df.iloc[:0]

    # Top up to n with a uniform sample from whatever is left.
    if len(pulled) < n:
        leftover = df.drop(pulled.index)
        if len(leftover):
            extra = leftover.sample(
                n=min(n - len(pulled), len(leftover)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            pulled = pd.concat([pulled, extra])

    return pulled.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def build_set_a_balanced(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Set A: balanced minors vs adults for general performance evaluation."""
    spec = cfg["test_sets"]["set_a_balanced"]
    minor_lo, minor_hi = spec["minor_age_range"]
    adult_lo, adult_hi = spec["adult_age_range"]
    n = spec["n_per_bin"]

    minors = df[(df["age"] >= minor_lo) & (df["age"] <= minor_hi)]
    adults = df[(df["age"] >= adult_lo) & (df["age"] <= adult_hi)]

    LOG.info("Set A pool sizes — minors: %d, adults: %d (target n_per_bin=%d)",
             len(minors), len(adults), n)
    if len(minors) < n:
        LOG.warning("Only %d minor samples available (asked for %d). Using all.", len(minors), n)
    if len(adults) < n:
        LOG.warning("Only %d adult samples available (asked for %d). Using all.", len(adults), n)

    minors_s = _stratified_sample(minors, n, by=["gender", "race"], rng=rng).copy()
    adults_s = _stratified_sample(adults, n, by=["gender", "race"], rng=rng).copy()
    minors_s["group"] = "minor"
    adults_s["group"] = "adult"

    out = pd.concat([minors_s, adults_s], ignore_index=True)
    out = out.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    out["sample_id"] = [f"A_{i:05d}" for i in range(len(out))]
    return out[["sample_id", "path", "filename", "age", "gender", "race", "group"]]


def build_set_b_minors(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Set B: minors only. Manipulations are applied later, in apply_manipulations.py."""
    spec = cfg["test_sets"]["set_b_minors_manipulated"]
    lo, hi = spec["age_range"]
    n = spec["n_samples"]

    pool = df[(df["age"] >= lo) & (df["age"] <= hi)]
    LOG.info("Set B pool size: %d (target n=%d)", len(pool), n)
    if len(pool) < n:
        LOG.warning("Only %d samples available for Set B (asked %d). Using all.", len(pool), n)

    sampled = _stratified_sample(pool, n, by=["gender", "race", "age"], rng=rng).copy()
    sampled["sample_id"] = [f"B_{i:05d}" for i in range(len(sampled))]
    sampled["group"] = "minor"
    return sampled[["sample_id", "path", "filename", "age", "gender", "race", "group"]]


def build_and_save(df: pd.DataFrame, cfg: dict) -> dict[str, Path]:
    """Build all enabled test sets and write them as CSV manifests.

    Returns a mapping from set name to manifest path.
    """
    paths = Paths.from_config(cfg)
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    manifests: dict[str, Path] = {}

    if cfg["test_sets"]["set_a_balanced"]["enabled"]:
        a = build_set_a_balanced(df, cfg, rng)
        out = paths.manifests_dir / "set_a_balanced.csv"
        a.to_csv(out, index=False)
        LOG.info("Wrote Set A manifest: %d rows -> %s", len(a), out)
        manifests["set_a"] = out

    if cfg["test_sets"]["set_b_minors_manipulated"]["enabled"]:
        b = build_set_b_minors(df, cfg, rng)
        out = paths.manifests_dir / "set_b_minors.csv"
        b.to_csv(out, index=False)
        LOG.info("Wrote Set B manifest: %d rows -> %s", len(b), out)
        manifests["set_b"] = out

    return manifests
