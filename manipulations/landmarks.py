"""Face landmark detection.

Tries multiple backends in order so that the framework runs across mediapipe versions:

  1. ``mediapipe.solutions.face_mesh`` — the classic API (mediapipe < 0.10 and some
     0.10.x builds). 478 landmarks (with refine_landmarks=True).

  2. ``mediapipe.tasks.python.vision.FaceLandmarker`` — the new Tasks API. Requires
     a downloaded ``.task`` model file. We auto-download to ``assets/weights/`` on
     first use if internet is available; otherwise this backend is skipped.

  3. ``dlib`` 68-point predictor as a last resort. If you want this fallback you must
     download ``shape_predictor_68_face_landmarks.dat`` and place it at
     ``assets/weights/shape_predictor_68_face_landmarks.dat`` (see assets/weights/README.md).

The module exposes a single function, ``detect_face_context(image_bgr)``, which always
returns a ``FaceContext`` (possibly with ``landmarks=None`` if every backend fails).

For consistency, we always normalize landmarks to MediaPipe's 478-index layout. When
running on dlib, we fill in only the indices we use elsewhere (defined in ``MP_INDICES``)
and leave the rest as zeros.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from manipulations.base import FaceContext
from utils import get_logger

LOG = get_logger(__name__)

# Indices into MediaPipe's 478-point FaceMesh that we use elsewhere in the codebase.
# (Stable across mediapipe 0.8 - 0.10.)
MP_INDICES = {
    "mouth_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    409, 270, 269, 267, 0, 37, 39, 40, 185],
    "upper_lip_top": [0, 37, 267, 269, 270, 409, 291, 61, 185, 40, 39],
    "nose_tip": 1,
    "nose_bottom": 2,
    "subnasal": 164,
    "chin": 152,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_brow": 105,
    "right_brow": 334,
    "forehead_top": 10,
}

# Where to look for / download model files.
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "weights"

# Public URL for the FaceLandmarker model (provided by Google).
MP_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
               "face_landmarker/float16/1/face_landmarker.task")
MP_TASK_PATH = WEIGHTS_DIR / "face_landmarker.task"


# --------------------------------------------------------------------------
# Backend 1: legacy mediapipe.solutions.face_mesh
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_legacy_facemesh():
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            return None
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
    except Exception as e:
        LOG.debug("Legacy mediapipe.solutions.face_mesh unavailable: %s", e)
        return None


def _detect_legacy(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    import cv2
    mesh = _get_legacy_facemesh()
    if mesh is None:
        return None
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


# --------------------------------------------------------------------------
# Backend 2: modern mediapipe Tasks API
# --------------------------------------------------------------------------

def _ensure_mp_task_file() -> Optional[Path]:
    """Make sure the FaceLandmarker .task file is on disk; download if missing."""
    if MP_TASK_PATH.exists():
        return MP_TASK_PATH
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        LOG.info("Downloading FaceLandmarker model from %s", MP_TASK_URL)
        urllib.request.urlretrieve(MP_TASK_URL, MP_TASK_PATH)
        return MP_TASK_PATH
    except Exception as e:
        LOG.warning("Could not download FaceLandmarker model: %s", e)
        return None


@lru_cache(maxsize=1)
def _get_tasks_landmarker():
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        LOG.debug("mediapipe Tasks API unavailable: %s", e)
        return None
    task_path = _ensure_mp_task_file()
    if task_path is None:
        return None
    try:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            output_face_blendshapes=False,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)
    except Exception as e:
        LOG.warning("Failed to init FaceLandmarker: %s", e)
        return None


def _detect_tasks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    landmarker = _get_tasks_landmarker()
    if landmarker is None:
        return None
    try:
        import cv2
        from mediapipe import Image, ImageFormat
    except Exception:
        return None
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


# --------------------------------------------------------------------------
# Backend 3: dlib 68-point fallback
# --------------------------------------------------------------------------

DLIB_TO_MP = {
    1: 30, 2: 33, 164: 33, 152: 8,
    33: 36, 133: 39, 362: 42, 263: 45,
    105: 19, 334: 24, 10: 27,
    61: 48, 291: 54,
}
DLIB_MOUTH_OUTER = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
DLIB_UPPER_LIP_TOP = [49, 50, 51, 52, 53]


@lru_cache(maxsize=1)
def _get_dlib_predictor():
    try:
        import dlib
    except ImportError:
        LOG.debug("dlib not installed, fallback unavailable.")
        return None, None
    detector = dlib.get_frontal_face_detector()
    model_path = WEIGHTS_DIR / "shape_predictor_68_face_landmarks.dat"
    if not model_path.exists():
        LOG.debug("dlib 68-point model not found at %s", model_path)
        return None, None
    try:
        predictor = dlib.shape_predictor(str(model_path))
    except Exception as e:
        LOG.warning("dlib predictor load failed: %s", e)
        return None, None
    return detector, predictor


def _detect_dlib(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    import cv2
    detector, predictor = _get_dlib_predictor()
    if detector is None or predictor is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
    shape = predictor(gray, rect)
    pts68 = np.array([[shape.part(i).x, shape.part(i).y]
                      for i in range(68)], dtype=np.float32)
    out = np.zeros((478, 2), dtype=np.float32)
    for mp_idx, dlib_idx in DLIB_TO_MP.items():
        out[mp_idx] = pts68[dlib_idx]
    mouth_mp_idx = MP_INDICES["mouth_outer"]
    for i, mp_idx in enumerate(mouth_mp_idx):
        out[mp_idx] = pts68[DLIB_MOUTH_OUTER[i % len(DLIB_MOUTH_OUTER)]]
    upper_lip_mp_idx = MP_INDICES["upper_lip_top"]
    for i, mp_idx in enumerate(upper_lip_mp_idx):
        out[mp_idx] = pts68[DLIB_UPPER_LIP_TOP[i % len(DLIB_UPPER_LIP_TOP)]]
    return out


# --------------------------------------------------------------------------
# Public entry-point
# --------------------------------------------------------------------------

_BACKEND_ORDER = (("legacy", _detect_legacy),
                  ("tasks", _detect_tasks),
                  ("dlib", _detect_dlib))


def detect_face_context(image_bgr: np.ndarray) -> FaceContext:
    """Detect a single face. Returns a FaceContext (possibly with landmarks=None)."""
    h, w = image_bgr.shape[:2]
    pts: Optional[np.ndarray] = None
    used: str = "none"
    for name, fn in _BACKEND_ORDER:
        try:
            pts = fn(image_bgr)
        except Exception as e:
            LOG.debug("Backend %s raised: %s", name, e)
            pts = None
        if pts is not None:
            used = name
            break

    if pts is None:
        return FaceContext(landmarks=None, bbox=None, image_shape=(h, w))

    nonzero = pts[(pts[:, 0] > 0) | (pts[:, 1] > 0)]
    if len(nonzero) == 0:
        return FaceContext(landmarks=None, bbox=None, image_shape=(h, w))
    x1, y1 = nonzero.min(axis=0)
    x2, y2 = nonzero.max(axis=0)
    bbox = (max(0, int(x1)), max(0, int(y1)),
            min(w, int(x2)), min(h, int(y2)))
    LOG.debug("Landmarks via backend=%s, %d points", used, len(pts))
    return FaceContext(landmarks=pts, bbox=bbox, image_shape=(h, w))


def get_index(landmarks: np.ndarray, key: str) -> np.ndarray:
    idx = MP_INDICES[key]
    if isinstance(idx, int):
        return landmarks[idx]
    return landmarks[idx]
