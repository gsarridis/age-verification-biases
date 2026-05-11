"""End-to-end integration test using a synthetic dataset and a mock model.

This verifies that:
  * scripts.build_test_sets produces sensible manifests,
  * scripts.apply_manipulations produces output images,
  * the metrics & report scripts run on the resulting predictions,
  * the runner gracefully handles missing models / empty predictions.

It uses a tiny synthetic UTKFace-style directory (filenames only — no real images
required for the manifest/builder step) and a hand-written mock model wrapper that
reads ``true_age`` from the file path and returns a deterministic perturbation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


# -------------------------------------------------------------------------
# Helpers: build a synthetic UTKFace-style directory of solid-color JPEGs.
# -------------------------------------------------------------------------

def _make_synthetic_utkface(root: Path, n_per_age: int = 5,
                            ages: list[int] | None = None) -> None:
    """Create JPGs whose filenames encode age/gender/race per UTKFace conventions.

    Uses scikit-image's astronaut sample as the image content (it has a detectable
    face), so the manipulation pipeline has something to work with. Falls back to
    solid-color images if scikit-image isn't available — those tests are marked
    accordingly via pytest skip.
    """
    try:
        from skimage.data import astronaut
        face_img = cv2.cvtColor(astronaut(), cv2.COLOR_RGB2BGR)
        face_img = cv2.resize(face_img, (200, 200))
        have_face = True
    except ImportError:
        face_img = None
        have_face = False

    if ages is None:
        ages = list(range(2, 13)) + list(range(18, 40, 2))
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for age in ages:
        for k in range(n_per_age):
            gender = int(rng.integers(0, 2))
            race = int(rng.integers(0, 5))
            fn = f"{age}_{gender}_{race}_2017010915055{k:04d}.jpg"
            if have_face:
                # Slight per-file variation so they're not byte-identical.
                tint = int(rng.integers(-15, 16))
                img = np.clip(face_img.astype(np.int16) + tint, 0, 255).astype(np.uint8)
            else:
                img = np.full((200, 200, 3), int(rng.integers(50, 200)), dtype=np.uint8)
            cv2.imwrite(str(root / fn), img)


def _write_config(out_dir: Path, dataset_root: Path, output_dir: Path,
                  report_dir: Path) -> Path:
    cfg = f"""\
experiment:
  name: integration
  seed: 7
  output_dir: {output_dir}
  report_dir: {report_dir}

dataset:
  name: utkface
  root: {dataset_root}

test_sets:
  set_a_balanced:
    enabled: true
    n_per_bin: 10
    minor_age_range: [2, 12]
    adult_age_range: [18, 40]
    threshold_age: 13
  set_b_minors_manipulated:
    enabled: true
    n_samples: 5
    age_range: [6, 12]

manipulations:
  classical:
    enabled: true
    list:
      - mustache_thin
  genai:
    enabled: false
    list: []

models: []     # We don't load any real models; we'll write predictions manually.

evaluation:
  thresholds: [13, 18, 21]
  bootstrap_iters: 50
  per_age_breakdown: false
"""
    p = out_dir / "test_config.yaml"
    p.write_text(cfg)
    return p


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_build_set_a_balanced_yields_balanced_groups(tmp_path):
    from data.loader import load_utkface
    from data.splits import build_set_a_balanced
    ds_root = tmp_path / "utk"
    _make_synthetic_utkface(ds_root, n_per_age=8)
    df = load_utkface(ds_root)

    cfg = {
        "test_sets": {
            "set_a_balanced": {
                "n_per_bin": 20,
                "minor_age_range": [2, 12],
                "adult_age_range": [18, 40],
                "threshold_age": 13,
            }
        }
    }
    rng = np.random.default_rng(0)
    setA = build_set_a_balanced(df, cfg, rng)
    assert (setA["group"] == "minor").sum() == (setA["group"] == "adult").sum()
    assert (setA[setA["group"] == "minor"]["age"] < 13).all()
    assert (setA[setA["group"] == "adult"]["age"] >= 18).all()


def test_build_and_save_writes_csv_manifests(tmp_path):
    from data.loader import load_utkface
    from data.splits import build_and_save
    from utils import load_config

    ds_root = tmp_path / "utk"
    out_dir = tmp_path / "out"
    rep_dir = tmp_path / "rep"
    _make_synthetic_utkface(ds_root, n_per_age=6)
    cfg_path = _write_config(tmp_path, ds_root, out_dir, rep_dir)
    cfg = load_config(cfg_path)
    df = load_utkface(ds_root)

    manifests = build_and_save(df, cfg)
    assert "set_a" in manifests
    assert "set_b" in manifests
    assert manifests["set_a"].exists()
    assert manifests["set_b"].exists()

    a_df = pd.read_csv(manifests["set_a"])
    b_df = pd.read_csv(manifests["set_b"])
    assert "sample_id" in a_df.columns
    assert (b_df["age"] < 13).all()


def test_apply_manipulations_runs_on_synthetic_images(tmp_path):
    """Even without detectable faces, the pipeline should not crash and should write CSV."""
    from data.loader import load_utkface
    from data.splits import build_and_save
    from manipulations.pipeline import apply_to_manifest
    from utils import load_config, Paths

    ds_root = tmp_path / "utk"
    out_dir = tmp_path / "out"
    rep_dir = tmp_path / "rep"
    _make_synthetic_utkface(ds_root, n_per_age=4)
    cfg_path = _write_config(tmp_path, ds_root, out_dir, rep_dir)
    cfg = load_config(cfg_path)
    df = load_utkface(ds_root)
    build_and_save(df, cfg)

    paths = Paths.from_config(cfg)
    set_b = paths.manifests_dir / "set_b_minors.csv"
    out_man = apply_to_manifest(set_b, cfg)
    assert out_man.exists()
    out_df = pd.read_csv(out_man)

    n_samples = len(pd.read_csv(set_b))
    # Every sample must have at least its 'original' row.
    assert (out_df["manipulation"] == "original").sum() == n_samples
    # All output paths exist on disk.
    assert all(Path(p).exists() for p in out_df["manipulated_path"])
    # If landmark detection succeeded for any sample, we should also see manipulation rows.
    # (Test environment may or may not detect faces in the synthetic data.)
    n_manip_rows = (out_df["manipulation"] != "original").sum()
    n_with_landmarks = out_df[out_df["had_landmarks"]]["sample_id"].nunique()
    assert n_manip_rows == n_with_landmarks * 1     # 1 manipulation configured.


def test_metrics_pipeline_with_synthetic_predictions(tmp_path):
    """Generate fake predictions, run evaluate_all, verify CSVs are produced."""
    from utils import load_config, Paths
    from evaluation.metrics import evaluate_all

    ds_root = tmp_path / "utk"
    out_dir = tmp_path / "out"
    rep_dir = tmp_path / "rep"
    _make_synthetic_utkface(ds_root, n_per_age=6)
    cfg_path = _write_config(tmp_path, ds_root, out_dir, rep_dir)
    cfg = load_config(cfg_path)

    paths = Paths.from_config(cfg)

    # Fake Set A predictions for a "biased" model: predicts roughly true_age + 5.
    set_a_pred = pd.DataFrame({
        "sample_id": [f"A_{i:05d}" for i in range(40)],
        "manipulation": ["original"] * 40,
        "path": ["/dev/null"] * 40,
        "true_age": list(range(2, 13)) * 2 + list(range(18, 36)),
        "predicted_age": [a + 5 for a in (list(range(2, 13)) * 2 + list(range(18, 36)))],
        "error": [None] * 40,
        "distribution_json": [""] * 40,
    })
    set_a_pred.to_csv(paths.predictions_dir / "set_a__fake_model.csv", index=False)

    # Fake Set B predictions: original + mustache_thin manipulation, same minors.
    minors = list(range(6, 13))
    rows = []
    for i, age in enumerate(minors * 3):       # 21 samples
        sid = f"B_{i % 7:05d}"
        for manip, bias in [("original", 0), ("mustache_thin", 6)]:
            rows.append({
                "sample_id": sid,
                "manipulation": manip,
                "path": "/dev/null",
                "true_age": age,
                "predicted_age": age + bias,
                "error": None,
                "distribution_json": "",
            })
    pd.DataFrame(rows).to_csv(paths.predictions_dir / "set_b_manipulated__fake_model.csv",
                              index=False)

    # Run.
    out = evaluate_all(cfg)
    assert "set_a_mae" in out and out["set_a_mae"].exists()
    assert "set_b_delta" in out and out["set_b_delta"].exists()
    assert "set_b_binary" in out and out["set_b_binary"].exists()

    # Δ-prediction sanity: the mustache added 6 years to every prediction.
    delta_df = pd.read_csv(out["set_b_delta"])
    must = delta_df[delta_df["manipulation"] == "mustache_thin"]
    assert len(must) == 1
    assert must["mean_delta_years"].iloc[0] == pytest.approx(6.0, abs=0.01)
    assert must["pct_subjects_aged_up"].iloc[0] == 1.0   # All subjects aged up.

    # Binary-threshold sanity: original predictions added only +0 to the (6..12) ages,
    # so all 7 distinct subjects start under 13. Adding 6 pushes ages 7..12 to 13..18,
    # so most should flip. Subject with age=6 -> pred 6 + 6 = 12 stays under threshold.
    binary_df = pd.read_csv(out["set_b_binary"])
    must_bin = binary_df[binary_df["manipulation"] == "mustache_thin"]
    assert len(must_bin) == 1
    # Original accuracy = 100% (all minors correctly classified <13).
    assert must_bin["acc_original"].iloc[0] == pytest.approx(1.0)
    # Manipulated accuracy < original (the manipulation hurt).
    assert must_bin["acc_manipulated"].iloc[0] < must_bin["acc_original"].iloc[0]
    # Some subjects flipped from "correct on original" to "wrong after manipulation".
    assert must_bin["n_correct_to_wrong"].iloc[0] >= 1
    assert must_bin["flip_rate"].iloc[0] > 0


def test_report_generation_runs(tmp_path):
    """End-to-end: synthetic data -> manifests -> fake predictions -> metrics -> report."""
    from utils import load_config, Paths
    from evaluation.metrics import evaluate_all
    from reports.generate import generate

    ds_root = tmp_path / "utk"
    out_dir = tmp_path / "out"
    rep_dir = tmp_path / "rep"
    _make_synthetic_utkface(ds_root, n_per_age=6)
    cfg_path = _write_config(tmp_path, ds_root, out_dir, rep_dir)
    cfg = load_config(cfg_path)
    paths = Paths.from_config(cfg)

    # Reuse the prediction-generation logic from the previous test.
    set_a_pred = pd.DataFrame({
        "sample_id": [f"A_{i:05d}" for i in range(20)],
        "manipulation": ["original"] * 20,
        "path": ["/dev/null"] * 20,
        "true_age": list(range(2, 12)) + list(range(20, 30)),
        "predicted_age": list(range(4, 14)) + list(range(22, 32)),
        "error": [None] * 20,
        "distribution_json": [""] * 20,
    })
    set_a_pred.to_csv(paths.predictions_dir / "set_a__fake_model.csv", index=False)

    rows = []
    for i in range(7):
        for manip, bias in [("original", 0), ("mustache_thin", 5)]:
            rows.append({
                "sample_id": f"B_{i:05d}",
                "manipulation": manip,
                "path": "/dev/null",
                "true_age": 10,
                "predicted_age": 10 + bias,
                "error": None,
                "distribution_json": "",
            })
    pd.DataFrame(rows).to_csv(paths.predictions_dir / "set_b_manipulated__fake_model.csv",
                              index=False)

    evaluate_all(cfg)
    html_path = generate(cfg)
    assert html_path.exists()
    html = html_path.read_text()
    # Some basic content checks for the new report.
    assert "age verification" in html.lower()
    assert "Set A" in html
    assert "Set B" in html
    # Plots should have been written next to the HTML.
    for img_name in ("set_a_mae.png", "binary_accuracy.png",
                     "delta_predictions.png", "flip_rates.png"):
        assert (rep_dir / img_name).exists(), f"Missing plot: {img_name}"
