"""Run all configured models over all test sets and write per-model prediction CSVs.

Per-model outputs:
  results/predictions/<set_name>__<model_name>.csv

with columns: sample_id, manipulation (or "original"), path, true_age, predicted_age,
              error, distribution_json.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# Importing wrappers registers them.
import models.deepface_wrapper  # noqa: F401
import models.insightface_wrapper  # noqa: F401
import models.hf_wrapper  # noqa: F401
import models.mivolo_wrapper  # noqa: F401
import models.fairface_wrapper  # noqa: F401
from models.base import AgePrediction, build_model
from utils import Paths, get_logger

LOG = get_logger(__name__)


def _iter_set(set_name: str, manifest_csv: Path):
    df = pd.read_csv(manifest_csv)
    if set_name == "set_b_manipulated":
        # Manifest already has one row per (sample, manipulation).
        for _, r in df.iterrows():
            yield r["sample_id"], r["manipulation"], r["manipulated_path"], r["age"]
    else:
        for _, r in df.iterrows():
            yield r["sample_id"], "original", r["path"], r["age"]


def _set_total(manifest_csv: Path) -> int:
    return len(pd.read_csv(manifest_csv))


def _serialize(p: AgePrediction) -> dict:
    return {
        "predicted_age": p.age,
        "error": p.error,
        "distribution_json": json.dumps(p.distribution) if p.distribution else "",
        **{f"extra_{k}": v for k, v in p.extra.items()},
    }


def run_model_on_set(
    model_cfg: dict, set_name: str, manifest_csv: Path, out_dir: Path
) -> Path:
    """Load a single model, run it over all rows in a manifest, save CSV."""
    backend = model_cfg["backend"]
    name = model_cfg["name"]
    out_path = out_dir / f"{set_name}__{name}.csv"

    if out_path.exists():
        LOG.info("Predictions already exist, skipping: %s", out_path)
        return out_path

    LOG.info("Loading model %s (backend=%s)…", name, backend)
    kwargs = {
        k: v for k, v in model_cfg.items() if k not in ("name", "backend", "enabled")
    }
    model = build_model(backend, **kwargs)
    # try:
    model.setup()
    # except Exception as e:
    #     LOG.error("Model %s failed setup: %s", name, e)
    #     return out_path

    rows: list[dict] = []
    total = _set_total(manifest_csv)
    for sample_id, manipulation, path, true_age in tqdm(
        _iter_set(set_name, manifest_csv),
        total=total,
        desc=f"{name} on {set_name}",
    ):
        img = cv2.imread(str(path))
        if img is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "manipulation": manipulation,
                    "path": str(path),
                    "true_age": true_age,
                    "predicted_age": None,
                    "error": "read_failed",
                    "distribution_json": "",
                }
            )
            continue
        pred = model.predict_age(img)
        rows.append(
            {
                "sample_id": sample_id,
                "manipulation": manipulation,
                "path": str(path),
                "true_age": true_age,
                **_serialize(pred),
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    LOG.info("Wrote %d predictions -> %s", len(rows), out_path)

    # Clean up to free GPU memory before the next model.
    try:
        model.teardown()
    except Exception:
        pass
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return out_path


def run_all(cfg: dict) -> dict[str, list[Path]]:
    """Run every enabled model over every existing manifest. Returns {set_name: [csv_paths]}."""
    paths = Paths.from_config(cfg)
    pred_dir = paths.predictions_dir

    # Discover available manifests.
    manifests: dict[str, Path] = {}
    set_a = paths.manifests_dir / "set_a_balanced.csv"
    set_b_man = paths.manifests_dir / "set_b_manipulated.csv"
    set_b_raw = paths.manifests_dir / "set_b_minors.csv"
    if set_a.exists():
        manifests["set_a"] = set_a
    if set_b_man.exists():
        manifests["set_b_manipulated"] = set_b_man
    elif set_b_raw.exists():
        # If manipulations have not yet been applied, we can still evaluate originals.
        LOG.warning(
            "No manipulated manifest found; using set_b_minors.csv (originals only)."
        )
        manifests["set_b_minors"] = set_b_raw

    enabled_models = [m for m in cfg["models"] if m.get("enabled", True)]
    LOG.info(
        "Running %d model(s) over %d manifest(s).", len(enabled_models), len(manifests)
    )

    out: dict[str, list[Path]] = {k: [] for k in manifests}
    for model_cfg in enabled_models:
        for set_name, manifest in manifests.items():
            try:
                p = run_model_on_set(model_cfg, set_name, manifest, pred_dir)
                out[set_name].append(p)
            except Exception as e:
                LOG.exception(
                    "Model %s failed on %s: %s", model_cfg["name"], set_name, e
                )
    return out
