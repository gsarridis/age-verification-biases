"""Unit tests for the pure-logic modules.

These tests do NOT require any face dataset or model downloads — they exercise:
  * config loading & override
  * UTKFace filename parsing
  * metric calculations
  * manipulation registry & overlay generation on synthetic images
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------- Config ----------

def test_load_config_expands_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_TEST_DIR", "/tmp/abc")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("dataset:\n  root: ${MY_TEST_DIR}\n  name: utkface\n")
    from utils import load_config
    out = load_config(cfg)
    assert out["dataset"]["root"] == "/tmp/abc"


def test_apply_overrides_scalar_and_nested():
    from utils import apply_overrides
    cfg = {"experiment": {"seed": 1}, "manipulations": {"genai": {"enabled": False}}}
    apply_overrides(cfg, ["experiment.seed=42", "manipulations.genai.enabled=true"])
    assert cfg["experiment"]["seed"] == 42
    assert cfg["manipulations"]["genai"]["enabled"] is True


def test_apply_overrides_creates_missing_nodes():
    from utils import apply_overrides
    cfg = {}
    apply_overrides(cfg, ["a.b.c=3.14"])
    assert cfg["a"]["b"]["c"] == 3.14


# ---------- UTKFace filename parsing ----------

def test_utkface_regex_matches_canonical():
    from data.loader import UTKFACE_NAME_RE
    m = UTKFACE_NAME_RE.match("7_0_3_20170109150557335.jpg.chip.jpg")
    assert m
    assert m.group(1) == "7"
    assert m.group(2) == "0"
    assert m.group(3) == "3"


def test_utkface_regex_rejects_garbage():
    from data.loader import UTKFACE_NAME_RE
    assert UTKFACE_NAME_RE.match("hello.jpg") is None
    assert UTKFACE_NAME_RE.match("7_X_3_blah.jpg") is None


def test_load_utkface_with_synthetic_dir(tmp_path):
    """Create empty files with valid filenames and verify parsing."""
    from data.loader import load_utkface
    # Create some valid filenames.
    for fn in [
        "5_0_0_20161219140622307.jpg",
        "10_1_2_20161219140623097.jpg",
        "30_0_3_20170109150557335.jpg.chip.jpg",
        "garbage.jpg",
    ]:
        # Need an actual readable image file? load_utkface only checks filename, not content.
        (tmp_path / fn).touch()
    df = load_utkface(tmp_path)
    assert len(df) == 3
    assert set(df["age"].tolist()) == {5, 10, 30}


# ---------- Metrics ----------

def test_mae():
    from evaluation.metrics import mae
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 22.0, 28.0])
    assert mae(y, p) == 2.0


def test_bootstrap_mean_ci_returns_finite():
    from evaluation.metrics import bootstrap_mean_ci
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=5.0, size=200)
    lo, hi = bootstrap_mean_ci(vals, n_iters=200, rng=rng)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    # The mean is ~5; the CI should bracket it loosely.
    assert lo < 5.0 < hi


def test_bootstrap_proportion_ci_returns_in_unit_interval():
    from evaluation.metrics import bootstrap_proportion_ci
    rng = np.random.default_rng(0)
    successes = (rng.random(200) < 0.4).astype(int)
    lo, hi = bootstrap_proportion_ci(successes, n_iters=200, rng=rng)
    assert 0.0 <= lo <= hi <= 1.0
    # The true proportion is 0.4; CI should bracket it.
    assert lo < 0.4 < hi


# ---------- Manipulation registry ----------

def test_classical_manipulations_registered():
    import manipulations.classical  # noqa: F401
    from manipulations.base import list_manipulations
    names = list_manipulations()
    for expected in ["mustache_thin", "mustache_thick", "beard_stubble",
                     "eye_makeup_heavy", "lipstick_red", "glasses_adult",
                     "hat_adult", "aging_wrinkles"]:
        assert expected in names, f"Missing: {expected}"


def test_overlay_generators_produce_valid_pngs():
    from manipulations.classical import (_make_mustache_stamp,
                                         _make_glasses_stamp, _make_hat_stamp)
    for fn, w in [(_make_mustache_stamp, 80),
                  (_make_glasses_stamp, 100),
                  (_make_hat_stamp, 200)]:
        stamp = fn(w)
        assert stamp.ndim == 3 and stamp.shape[2] == 4
        assert stamp.dtype == np.uint8
        assert stamp[..., 3].max() > 0   # Has some non-transparent pixels.


def test_alpha_blit_clips_out_of_bounds():
    from manipulations.classical import _alpha_blit
    dst = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = np.full((6, 6, 4), 255, dtype=np.uint8)
    _alpha_blit(dst, overlay, top_left=(8, 8))   # Mostly off-canvas.
    # Should not crash and should have written to the 2x2 corner only.
    assert dst[8:10, 8:10].max() == 255
    assert dst[0:8, 0:8].max() == 0


# ---------- Integration: classical manipulation on a synthetic image ----------

def test_classical_pipeline_on_synthetic_image_no_face():
    """When no landmarks are detected, the manipulation should return the input unchanged."""
    from manipulations.base import FaceContext
    from manipulations.classical import MustacheThin
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    ctx = FaceContext(landmarks=None, bbox=None, image_shape=(100, 100))
    out = MustacheThin().apply(img, ctx)
    assert (out == img).all()
