"""Pipeline integration test.

Runs the synthetic smoke test through the same code path the real pipeline uses.
Skipped automatically if mediapipe / scikit-image are unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_smoke_pipeline_end_to_end(tmp_path):
    """Exercise build → manipulate → fake-predict → evaluate → report end to end."""
    pytest.importorskip("mediapipe")
    pytest.importorskip("skimage")

    # Run the smoke script as a subprocess so we don't pollute pytest's import state.
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Smoke test failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # Verify key artifacts were produced.
    smoke_dir = PROJECT_ROOT / "outputs" / "smoke"
    assert (smoke_dir / "results" / "manifests" / "set_a_balanced.csv").exists()
    assert (smoke_dir / "results" / "manifests" / "set_b_manipulated.csv").exists()
    assert (smoke_dir / "results" / "metrics" / "set_a_mae.csv").exists()
    assert (smoke_dir / "outputs" / "report.html").exists()
    # Δ-prediction and binary-threshold CSVs only appear when face detection succeeded
    # on Set B images; otherwise the manipulation rows are missing. We don't require them
    # here so the test doesn't depend on the local environment's mediapipe version.


def test_pipeline_module_imports():
    """All package modules import cleanly without optional deps."""
    import data.loader      # noqa: F401
    import data.splits      # noqa: F401
    import evaluation.metrics  # noqa: F401
    import manipulations.base  # noqa: F401
    import manipulations.classical  # noqa: F401
    import models.base      # noqa: F401
