"""Tests for model wrapper logic that don't require model weight downloads.

These tests exercise:
  * The bucket-midpoint expectation math used by HFAgeClassifier and FairFaceAge.
  * The AgeModel registry & build_model dispatch.
  * The AgePrediction dataclass serialization shape.

The actual model loading & prediction is gated behind ``setup()`` and tested separately
in ``test_models_integration.py`` — that file requires network access and model weights.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_all_model_backends_register():
    # Importing all wrapper modules registers the corresponding backends.
    import models.deepface_wrapper  # noqa: F401
    import models.insightface_wrapper  # noqa: F401
    import models.hf_wrapper  # noqa: F401
    import models.mivolo_wrapper  # noqa: F401
    import models.fairface_wrapper  # noqa: F401
    from models.base import list_backends

    backends = list_backends()
    for expected in ("deepface", "insightface", "hf_transformers", "mivolo", "fairface"):
        assert expected in backends, f"Missing backend: {expected}"


def test_build_model_returns_correct_class():
    import models.hf_wrapper  # noqa: F401
    from models.base import build_model
    from models.hf_wrapper import HFAgeClassifier

    m = build_model("hf_transformers", model_id="dummy/foo")
    assert isinstance(m, HFAgeClassifier)
    assert m.model_id == "dummy/foo"


def test_build_model_unknown_backend_raises():
    from models.base import build_model
    with pytest.raises(KeyError):
        build_model("does_not_exist")


# ---------------------------------------------------------------------------
# Bucket-midpoint expectation math
# ---------------------------------------------------------------------------

def test_hf_expected_age_singleton():
    from models.hf_wrapper import HFAgeClassifier
    id2label = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
                5: "40-49", 6: "50-59", 7: "60-69", 8: "more than 70"}
    probs = np.zeros(9)
    probs[3] = 1.0
    exp, _ = HFAgeClassifier._expected_age(probs, id2label)
    assert exp == 24.5


def test_hf_expected_age_bimodal():
    from models.hf_wrapper import HFAgeClassifier
    id2label = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
                5: "40-49", 6: "50-59", 7: "60-69", 8: "more than 70"}
    probs = np.zeros(9)
    probs[0] = 0.5
    probs[7] = 0.5
    exp, dist = HFAgeClassifier._expected_age(probs, id2label)
    assert exp == (1.0 + 64.5) / 2
    assert dist["0-2"] == 0.5
    assert dist["60-69"] == 0.5


def test_hf_expected_age_handles_custom_buckets():
    """If the id2label map uses non-standard names, parse 'lo-hi' format."""
    from models.hf_wrapper import HFAgeClassifier
    id2label = {0: "5-10", 1: "11-20"}
    probs = np.array([1.0, 0.0])
    exp, _ = HFAgeClassifier._expected_age(probs, id2label)
    assert exp == 7.5


# ---------------------------------------------------------------------------
# AgePrediction dataclass
# ---------------------------------------------------------------------------

def test_age_prediction_default_extra_is_dict():
    from models.base import AgePrediction
    p = AgePrediction(age=24.5)
    assert p.extra == {}
    assert p.distribution is None
    assert p.error is None


def test_age_prediction_serializable_fields():
    from models.base import AgePrediction
    from evaluation.runner import _serialize
    import json

    p = AgePrediction(age=10.5, distribution={"a": 0.5, "b": 0.5},
                      error=None, extra={"gender": 0})
    s = _serialize(p)
    assert s["predicted_age"] == 10.5
    assert s["error"] is None
    assert json.loads(s["distribution_json"]) == {"a": 0.5, "b": 0.5}
    assert s["extra_gender"] == 0


def test_age_prediction_error_case():
    from models.base import AgePrediction
    from evaluation.runner import _serialize
    p = AgePrediction(age=None, error="no_face")
    s = _serialize(p)
    assert s["predicted_age"] is None
    assert s["error"] == "no_face"
    assert s["distribution_json"] == ""


# ---------------------------------------------------------------------------
# FairFace bucket math (uses the same midpoints as HF, different label format)
# ---------------------------------------------------------------------------

def test_fairface_bucket_midpoints_length():
    from models.fairface_wrapper import _BUCKET_MIDPOINTS, _BUCKET_LABELS
    assert len(_BUCKET_MIDPOINTS) == len(_BUCKET_LABELS) == 9
