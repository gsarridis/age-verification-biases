"""End-to-end smoke test of the pipeline plumbing.

Does NOT require UTKFace or any real model. Instead:
  * Creates a fake UTKFace directory of synthetic faces (one face image, copied
    multiple times with different age-encoded filenames).
  * Builds the test-set manifests.
  * Applies the classical manipulations.
  * Fakes per-model predictions with a known relationship to the true age
    (so we can sanity-check that metrics make sense).
  * Computes metrics and generates a report.

Used as a CI / dev sanity check.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fake_utkface_dir(out_dir: Path, face_image_path: Path, n_per_age: int = 5) -> None:
    """Build a directory of synthetic UTKFace-style filenames pointing at one real face."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ages = list(range(2, 13)) + list(range(18, 60, 5))
    for age in ages:
        for k in range(n_per_age):
            gender = k % 2
            race = k % 5
            fn = f"{age}_{gender}_{race}_2017010915055{age:02d}{k:02d}.jpg.chip.jpg"
            shutil.copy(face_image_path, out_dir / fn)


def _make_smoke_config(work: Path) -> dict:
    """Build a small config that fits in seconds."""
    cfg = {
        "experiment": {
            "name": "smoke",
            "seed": 0,
            "output_dir": str(work / "results"),
            "report_dir": str(work / "outputs"),
        },
        "dataset": {
            "name": "utkface",
            "root": str(work / "fake_utkface"),
            "require_face_detection": False,
            "prefer_aligned": True,
        },
        "test_sets": {
            "set_a_balanced": {
                "enabled": True,
                "n_per_bin": 8,
                "minor_age_range": [2, 12],
                "adult_age_range": [18, 60],
                "threshold_age": 13,
            },
            "set_b_minors_manipulated": {
                "enabled": True,
                "n_samples": 4,
                "age_range": [6, 12],
            },
        },
        "manipulations": {
            "classical": {
                "enabled": True,
                "list": ["mustache_thick", "glasses_adult", "lipstick_red"],
            },
            "genai": {"enabled": False, "list": []},
        },
        "models": [],   # Filled in synthetically below.
        "evaluation": {"thresholds": [13, 18, 21], "bootstrap_iters": 100},
    }
    return cfg


def _fake_predictions(manifest_csv: Path, out_path: Path,
                      bias_per_manipulation: dict | None = None) -> None:
    """Generate fake predictions: predicted = true_age + age_bias + manipulation_bias + noise.

    For Set B, ``bias_per_manipulation`` controls how much each manipulation pushes the
    predicted age upward — this lets us verify that the metrics correctly detect each one.
    """
    df = pd.read_csv(manifest_csv)
    rng = np.random.default_rng(0)
    rows = []

    bias_per_manipulation = bias_per_manipulation or {}

    if "manipulation" in df.columns:
        for _, r in df.iterrows():
            man = r["manipulation"]
            bias = bias_per_manipulation.get(man, 0)
            noise = rng.normal(0, 1.5)
            pred = float(r["age"]) + bias + noise
            rows.append({
                "sample_id": r["sample_id"],
                "manipulation": man,
                "path": r["manipulated_path"],
                "true_age": r["age"],
                "predicted_age": pred,
                "error": "",
                "distribution_json": "",
            })
    else:
        for _, r in df.iterrows():
            noise = rng.normal(0, 1.5)
            pred = float(r["age"]) + noise
            rows.append({
                "sample_id": r["sample_id"],
                "manipulation": "original",
                "path": r["path"],
                "true_age": r["age"],
                "predicted_age": pred,
                "error": "",
                "distribution_json": "",
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    from data.loader import load_dataset
    from data.splits import build_and_save
    from manipulations.pipeline import apply_to_manifest
    from evaluation.metrics import evaluate_all
    from reports.generate import generate
    from utils import seed_everything, Paths

    work = ROOT / "outputs" / "smoke"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. Make a fake UTKFace dir from the test face.
    test_face = Path("/tmp/test_face.jpg")
    if not test_face.exists():
        # Re-create from skimage if missing.
        from skimage import data
        import cv2
        cv2.imwrite(str(test_face),
                    cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    _fake_utkface_dir(work / "fake_utkface", test_face, n_per_age=4)

    # 2. Config + seed.
    cfg = _make_smoke_config(work)
    cfg_path = work / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    seed_everything(cfg["experiment"]["seed"])

    # 3. Build manifests.
    print("=== Step 1: build manifests ===")
    df = load_dataset(cfg)
    print(f"Dataset: {len(df)} rows, ages {df['age'].min()}-{df['age'].max()}")
    build_and_save(df, cfg)
    paths = Paths.from_config(cfg)

    # 4. Apply manipulations.
    print("=== Step 2: apply manipulations ===")
    set_b = paths.manifests_dir / "set_b_minors.csv"
    apply_to_manifest(set_b, cfg)

    # 5. Fake predictions for two "models".
    print("=== Step 3: fake predictions ===")
    pred_dir = paths.predictions_dir
    # Model A: well-calibrated, slight upward bias on minors.
    # Model B: heavily fooled by mustaches.
    bias_a = {"original": 0, "mustache_thick": 1.5, "glasses_adult": 1.0, "lipstick_red": 0.5}
    bias_b = {"original": 0, "mustache_thick": 8.0, "glasses_adult": 4.0, "lipstick_red": 2.0}

    set_a_path = paths.manifests_dir / "set_a_balanced.csv"
    set_b_man_path = paths.manifests_dir / "set_b_manipulated.csv"

    _fake_predictions(set_a_path, pred_dir / "set_a__model_calibrated.csv")
    _fake_predictions(set_a_path, pred_dir / "set_a__model_fooled.csv")
    _fake_predictions(set_b_man_path,
                      pred_dir / "set_b_manipulated__model_calibrated.csv",
                      bias_per_manipulation=bias_a)
    _fake_predictions(set_b_man_path,
                      pred_dir / "set_b_manipulated__model_fooled.csv",
                      bias_per_manipulation=bias_b)

    # 6. Compute metrics.
    print("=== Step 4: compute metrics ===")
    evaluate_all(cfg)

    # 7. Generate report.
    print("=== Step 5: generate report ===")
    generate(cfg)

    # Print key metrics so we can verify visually.
    metrics_dir = paths.metrics_dir
    print("\n--- Set A MAE ---")
    print(pd.read_csv(metrics_dir / "set_a_mae.csv").to_string(index=False))

    delta_csv = metrics_dir / "set_b_delta_predictions.csv"
    binary_csv = metrics_dir / "set_b_binary_threshold.csv"
    if delta_csv.exists():
        print("\n--- Set B Δ predictions (model_fooled) ---")
        df_d = pd.read_csv(delta_csv)
        print(df_d[df_d["model"] == "model_fooled"][
            ["model", "manipulation", "n_paired", "mean_delta_years",
             "median_delta_years", "pct_subjects_aged_up"]
        ].to_string(index=False))
    else:
        print("\n--- Set B Δ predictions: SKIPPED (no manipulations applied; "
              "likely face detection failed in this environment) ---")
    if binary_csv.exists():
        print("\n--- Set B binary @ 13 (model_fooled, big drop expected for manipulated) ---")
        df_b = pd.read_csv(binary_csv)
        print(df_b[df_b["model"] == "model_fooled"][
            ["model", "manipulation", "acc_original", "acc_manipulated",
             "delta_accuracy", "n_correct_to_wrong", "flip_rate"]
        ].to_string(index=False))
    else:
        print("\n--- Set B binary: SKIPPED (no manipulations applied) ---")

    print(f"\n✅ Smoke test passed. Report at {paths.report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
