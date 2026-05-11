"""Classical-CV manipulations: deterministic overlay & color edits driven by landmarks.

These are NOT photorealistic, but they are:
  * fully reproducible (no model randomness),
  * fast (no GPU required),
  * landmark-aligned (so they sit in roughly the right place on each face).

For more realistic edits, see ``manipulations.genai``.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from manipulations.base import FaceContext, Manipulation, register
from manipulations.landmarks import MP_INDICES


# ============================================================
# Helpers
# ============================================================

def _safe_landmarks(ctx: FaceContext) -> np.ndarray | None:
    """Return landmarks if usable, else None.

    We require the array to be at least 478 rows AND that a few core indices we use
    (eye corners, mouth corners, nose) are non-zero. This works for both the dense
    MediaPipe path (all rows populated) and the sparse dlib fallback (only certain
    indices populated).
    """
    lms = ctx.landmarks
    if lms is None or len(lms) < 478:
        return None
    required = [33, 263, 61, 291, 1, 152]   # eyes outer, mouth corners, nose tip, chin
    for idx in required:
        if lms[idx, 0] == 0 and lms[idx, 1] == 0:
            return None
    return lms


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around the image center, expanding the canvas to avoid clipping."""
    h, w = img.shape[:2]
    cX, cY = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cX
    M[1, 2] += new_h / 2 - cY
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _alpha_blit(dst_bgr: np.ndarray, overlay_bgra: np.ndarray, top_left: tuple[int, int]) -> None:
    """In-place alpha-blend ``overlay_bgra`` onto ``dst_bgr`` at ``top_left`` (x, y).

    ``overlay_bgra`` must be BGRA uint8. Out-of-bounds regions are clipped.
    """
    x, y = top_left
    H, W = dst_bgr.shape[:2]
    h, w = overlay_bgra.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    region = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    over = overlay_bgra[oy0:oy1, ox0:ox1].astype(np.float32)
    alpha = (over[..., 3:4] / 255.0)
    blended = region * (1 - alpha) + over[..., :3] * alpha
    dst_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)


# ============================================================
# Procedural overlay generators (no PNG assets required).
# Generating the "stamps" procedurally keeps the framework
# self-contained and reproducible.
# ============================================================

def _make_mustache_stamp(width: int, thickness: float = 1.0,
                         color: tuple[int, int, int] = (15, 15, 15)) -> np.ndarray:
    """Return a BGRA stamp of a handlebar-style mustache scaled to ``width`` pixels.

    Geometry:
      * A horizontal "body" centered just below the nose.
      * A V-notch in the middle-bottom for the philtrum cleft.
      * Tips that curve slightly upward on each end.
      * Slight upward arc on the top edge so the mustache hugs the upper lip.
    """
    h = max(14, int(width * 0.45 * thickness))
    w = width
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    cx = w / 2.0

    # Vertical anchors.
    top_y_center = h * 0.30                # Top edge at the philtrum (sits under nose).
    top_y_tips = h * 0.10                  # Tips arc up.
    body_thickness = h * 0.45 * thickness  # How tall the body of the mustache is.
    notch_depth = h * 0.18                 # How far the V-notch cuts up into the body.
    bottom_y_tip = top_y_tips + body_thickness * 0.6   # Tips are a bit thinner.

    n = 60
    ts = np.linspace(-1.0, 1.0, n)         # -1 = left tip, 0 = center, +1 = right tip.

    # TOP edge: gentle upward smile away from the center.
    # y = top_y_center at t=0, rising to top_y_tips at |t|=1.
    top_pts = []
    for t in ts:
        # Smooth interpolation that's almost flat in the middle and rises near the tips.
        rise = (t ** 2) * (top_y_center - top_y_tips)   # 0 at center, full at tips.
        # Tips also curl up extra at the very ends.
        if abs(t) > 0.85:
            extra = ((abs(t) - 0.85) / 0.15) * (h * 0.10)
            rise += extra
        x = cx + t * (w / 2 - 2)
        y = top_y_center - rise
        top_pts.append([x, y])

    # BOTTOM edge: starts at top_y_center + body_thickness in the body, V-notches at center,
    # and approaches bottom_y_tip at the tips.
    bot_pts = []
    for t in ts:
        # Thickness profile: thick in the body, tapering at the tips.
        body_y = top_y_center + body_thickness * (1.0 - 0.4 * (t ** 2))
        # V-notch in the very middle (|t| < 0.18 lifts the bottom upward).
        if abs(t) < 0.18:
            # Triangular notch: at t=0, lifts by notch_depth; linearly decays to 0 at |t|=0.18.
            lift = notch_depth * (1 - abs(t) / 0.18)
            body_y -= lift
        # At the tips, taper toward bottom_y_tip.
        if abs(t) > 0.75:
            taper = ((abs(t) - 0.75) / 0.25)
            body_y = body_y * (1 - taper) + bottom_y_tip * taper
        x = cx + t * (w / 2 - 2)
        bot_pts.append([x, body_y])

    poly = np.array(top_pts + bot_pts[::-1], dtype=np.int32)
    cv2.fillPoly(canvas, [poly], (color[0], color[1], color[2], 255))

    # Soften the silhouette edges (alpha only — keeps the dark hair color crisp).
    alpha = canvas[..., 3]
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1.2)
    canvas[..., 3] = alpha

    # Hair-strand texture: short dark strokes inside the silhouette, deterministic.
    rng = np.random.default_rng(0)
    n_strands = int(w * thickness * 0.9)
    dark_color = (max(0, color[0] - 15), max(0, color[1] - 15), max(0, color[2] - 15), 255)
    for _ in range(n_strands):
        x0 = int(rng.integers(2, w - 2))
        y0 = int(rng.integers(2, h - 2))
        if canvas[y0, x0, 3] < 200:
            continue
        # Strands point outward from the center (away from philtrum).
        side = -1 if x0 < cx else 1
        dx = int(side * rng.integers(1, 5))
        dy = int(rng.integers(-2, 3))
        cv2.line(canvas, (x0, y0), (x0 + dx, y0 + dy), dark_color, 1, cv2.LINE_AA)
    return canvas


def _make_glasses_stamp(width: int, color: tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
    h = max(20, int(width * 0.35))
    w = width
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    lens_r = int(min(h * 0.45, w * 0.22))
    cy = h // 2
    left_cx = int(w * 0.28)
    right_cx = int(w * 0.72)

    # Two lens rings.
    cv2.circle(canvas, (left_cx, cy), lens_r, (*color, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (right_cx, cy), lens_r, (*color, 255), 3, lineType=cv2.LINE_AA)
    # Bridge.
    cv2.line(canvas, (left_cx + lens_r, cy), (right_cx - lens_r, cy),
             (*color, 255), 3, lineType=cv2.LINE_AA)
    # Temples (the side arms).
    cv2.line(canvas, (0, cy), (left_cx - lens_r, cy), (*color, 255), 3, lineType=cv2.LINE_AA)
    cv2.line(canvas, (right_cx + lens_r, cy), (w - 1, cy), (*color, 255), 3, lineType=cv2.LINE_AA)
    return canvas


def _make_hat_stamp(width: int) -> np.ndarray:
    """Simple fedora silhouette."""
    h = max(40, int(width * 0.6))
    w = width
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    color = (20, 20, 25, 255)
    # Brim: wide ellipse.
    cv2.ellipse(canvas, (w // 2, int(h * 0.78)), (int(w * 0.48), int(h * 0.13)),
                0, 0, 360, color, -1, lineType=cv2.LINE_AA)
    # Crown: rounded rectangle.
    crown_pts = np.array([
        [int(w * 0.25), int(h * 0.78)],
        [int(w * 0.27), int(h * 0.20)],
        [int(w * 0.35), int(h * 0.10)],
        [int(w * 0.65), int(h * 0.10)],
        [int(w * 0.73), int(h * 0.20)],
        [int(w * 0.75), int(h * 0.78)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [crown_pts], color)
    # Hat band.
    cv2.rectangle(canvas, (int(w * 0.27), int(h * 0.65)),
                  (int(w * 0.73), int(h * 0.74)), (60, 40, 25, 255), -1)
    return canvas


# ============================================================
# Manipulations
# ============================================================

class _MustacheBase(Manipulation):
    """Shared logic for mustache variants."""

    thickness: float = 1.0
    color: tuple[int, int, int] = (12, 12, 12)

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        # Anchor: between the bottom of the nose and the top of the upper lip.
        nose_bot = lms[MP_INDICES["nose_bottom"]]
        upper_lip = lms[MP_INDICES["upper_lip_top"]]
        # Average of the upper-lip ring gives the lip top center.
        lip_top = upper_lip.mean(axis=0)

        # Place center at 60% of the way from lip top up toward the nose.
        center = lip_top + 0.6 * (nose_bot - lip_top)

        # Width: distance between mouth corners times a factor.
        mouth_left = lms[61]
        mouth_right = lms[291]
        mouth_w = float(np.linalg.norm(mouth_right - mouth_left))
        if mouth_w < 5:
            return image.copy()
        stamp_w = int(mouth_w * 1.15)

        stamp = _make_mustache_stamp(stamp_w, thickness=self.thickness, color=self.color)

        # Rotate to match face roll.
        eye_l = lms[MP_INDICES["left_eye_outer"]]
        eye_r = lms[MP_INDICES["right_eye_outer"]]
        roll_rad = math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])
        stamp = _rotate_image(stamp, math.degrees(-roll_rad))

        out = image.copy()
        sh, sw = stamp.shape[:2]
        top_left = (int(center[0] - sw / 2), int(center[1] - sh / 2))
        _alpha_blit(out, stamp, top_left)
        return out


@register
class MustacheThin(_MustacheBase):
    name = "mustache_thin"
    thickness = 0.6


@register
class MustacheThick(_MustacheBase):
    name = "mustache_thick"
    thickness = 1.4


@register
class BeardStubble(Manipulation):
    """Stubble darkening over the lower face (jaw, chin, cheeks below cheekbones).

    The visible effect combines three layers:
      * a soft mask-bounded uniform darkening of the skin tone,
      * a sparse "follicle" speckle (small dark dots),
      * short directional hair-strand strokes pointing toward the chin.
    """
    name = "beard_stubble"

    # Tunables.
    base_darken: float = 22.0       # Uniform brightness reduction (0-255 scale).
    speckle_strength: float = 70.0  # Per-pixel darkening for the densest speckle.
    n_follicles_per_kpx: int = 5    # Density of explicit hair-follicle marks.

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        h, w = image.shape[:2]

        # Build the lower-face polygon. We use a wider arc that goes up along the jaw
        # to roughly the bottom of the ears, then across to just below each nostril and
        # the upper lip — covering the realistic stubble area.
        jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                       152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        jaw_pts = lms[jaw_indices].astype(np.int32)

        # Upper boundary: along the upper lip and out to the cheek/sideburn area.
        # 234 / 454 are roughly at the ear-top level on the jaw side already; we want the
        # closing line to come up just under the nose to include a mustache-area shadow.
        upper_lip_top = lms[MP_INDICES["upper_lip_top"]].mean(axis=0)
        nose_bot = lms[MP_INDICES["nose_bottom"]]
        # Two anchor points just outside each nostril, slightly above the lip.
        left_anchor = np.array([upper_lip_top[0] - (nose_bot[0] - upper_lip_top[0]) - 6,
                                upper_lip_top[1] - 4])
        right_anchor = np.array([upper_lip_top[0] + (upper_lip_top[0] - nose_bot[0]) + 6,
                                 upper_lip_top[1] - 4])
        # Build the polygon: jaw arc (left ear -> chin -> right ear),
        # then up to right cheekbone area, across under the nose, down to left cheekbone.
        poly = np.vstack([
            jaw_pts,
            right_anchor[None, :].astype(np.int32),
            np.array([upper_lip_top[0] + 6, upper_lip_top[1] - 1], dtype=np.int32)[None, :],
            np.array([upper_lip_top[0] - 6, upper_lip_top[1] - 1], dtype=np.int32)[None, :],
            left_anchor[None, :].astype(np.int32),
        ])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        # Exclude the lips themselves (don't darken the mouth).
        lip_outer = lms[MP_INDICES["mouth_outer"]].astype(np.int32)
        cv2.fillPoly(mask, [lip_outer], 0)

        # Soft falloff at the mask edge.
        mask_soft = cv2.GaussianBlur(mask, (25, 25), 8).astype(np.float32) / 255.0
        mask3 = mask_soft[..., None]

        out = image.astype(np.float32)

        # Layer 1: uniform darkening tinted slightly toward green-grey (skin under stubble).
        out -= self.base_darken * mask3

        # Layer 2: dense fine speckle (the "5-o-clock-shadow" texture).
        rng = np.random.default_rng(42)
        noise = rng.random(size=(h, w)).astype(np.float32)
        # ~30% of pixels get full speckle, ~25% get half, rest get nothing.
        speckle = np.where(noise < 0.30, -self.speckle_strength,
                  np.where(noise < 0.55, -self.speckle_strength * 0.45, 0.0)).astype(np.float32)
        out += (speckle * mask_soft)[..., None]

        # Layer 3: explicit short hair-strand strokes pointing toward the chin.
        out_uint = np.clip(out, 0, 255).astype(np.uint8)
        chin = lms[MP_INDICES["chin"]]
        # Sample stroke locations uniformly within the mask.
        ys, xs = np.where(mask > 200)
        if len(xs):
            n = max(40, int(len(xs) * self.n_follicles_per_kpx / 1000))
            idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
            for k in idx:
                px, py = int(xs[k]), int(ys[k])
                # Strand direction: roughly downward, biased slightly toward the chin.
                dx_to_chin = chin[0] - px
                dy_to_chin = chin[1] - py
                norm = max(1.0, math.hypot(dx_to_chin, dy_to_chin))
                length = float(rng.integers(2, 5))
                ex = int(px + (dx_to_chin / norm) * length * 0.4)
                ey = int(py + (dy_to_chin / norm) * length + rng.integers(-1, 2))
                cv2.line(out_uint, (px, py), (ex, ey),
                         (15, 18, 22), 1, cv2.LINE_AA)

        return out_uint


@register
class EyeMakeupHeavy(Manipulation):
    """Darkened eye-shadow + eyeliner around both eyes."""
    name = "eye_makeup_heavy"

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        out = image.copy()
        # Approximate eye + eyelid contours from MediaPipe.
        left_eye_ring = [33, 246, 161, 160, 159, 158, 157, 173, 133,
                         155, 154, 153, 145, 144, 163, 7]
        right_eye_ring = [263, 466, 388, 387, 386, 385, 384, 398, 362,
                          382, 381, 380, 374, 373, 390, 249]
        for ring in (left_eye_ring, right_eye_ring):
            pts = lms[ring].astype(np.int32)
            # Expand the ring slightly upward to cover the eyelid.
            center = pts.mean(axis=0)
            expanded = (pts - center) * 1.45 + center + np.array([0, -2])
            expanded = expanded.astype(np.int32)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [expanded], 255)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask3 = (mask.astype(np.float32) / 255.0)[..., None]

            # Smoky-eye color: dark plum/grey.
            shadow = np.full_like(image, fill_value=(45, 25, 35), dtype=np.uint8)
            out = (out.astype(np.float32) * (1 - 0.55 * mask3) +
                   shadow.astype(np.float32) * (0.55 * mask3)).astype(np.uint8)

            # Eyeliner: thin dark line just below the upper eyelid.
            top_half = pts[len(pts) // 4: 3 * len(pts) // 4]
            for i in range(len(top_half) - 1):
                cv2.line(out, tuple(top_half[i]), tuple(top_half[i + 1]),
                         (10, 10, 10), 2, cv2.LINE_AA)
        return out


@register
class LipstickRed(Manipulation):
    """Saturated red overlay on the lips."""
    name = "lipstick_red"

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        lip_outer = lms[MP_INDICES["mouth_outer"]].astype(np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [lip_outer], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 1)
        mask3 = (mask.astype(np.float32) / 255.0)[..., None]

        red = np.full_like(image, fill_value=(40, 20, 170), dtype=np.uint8)  # BGR red
        out = (image.astype(np.float32) * (1 - 0.65 * mask3) +
               red.astype(np.float32) * (0.65 * mask3)).astype(np.uint8)
        return out


@register
class GlassesAdult(Manipulation):
    """Dark-rimmed adult-style glasses."""
    name = "glasses_adult"

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        eye_l = lms[MP_INDICES["left_eye_outer"]]
        eye_r = lms[MP_INDICES["right_eye_outer"]]
        eye_w = float(np.linalg.norm(eye_r - eye_l))
        if eye_w < 5:
            return image.copy()

        stamp_w = int(eye_w * 2.4)
        stamp = _make_glasses_stamp(stamp_w)
        roll = math.degrees(math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
        stamp = _rotate_image(stamp, -roll)

        center = (eye_l + eye_r) / 2
        sh, sw = stamp.shape[:2]
        out = image.copy()
        _alpha_blit(out, stamp, (int(center[0] - sw / 2), int(center[1] - sh / 2)))
        return out


@register
class HatAdult(Manipulation):
    """Adds a fedora-like hat above the forehead."""
    name = "hat_adult"

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        forehead = lms[MP_INDICES["forehead_top"]]
        eye_l = lms[MP_INDICES["left_eye_outer"]]
        eye_r = lms[MP_INDICES["right_eye_outer"]]
        face_w = float(np.linalg.norm(eye_r - eye_l)) * 2.6
        if face_w < 20:
            return image.copy()

        stamp = _make_hat_stamp(int(face_w))
        roll = math.degrees(math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))
        stamp = _rotate_image(stamp, -roll)

        sh, sw = stamp.shape[:2]
        # Anchor: brim sits just at the forehead anchor; the rest of the hat extends upward.
        top_left = (int(forehead[0] - sw / 2), int(forehead[1] - sh * 0.78))
        out = image.copy()
        _alpha_blit(out, stamp, top_left)
        return out


@register
class AgingWrinkles(Manipulation):
    """Subtle wrinkle-like darkening around eyes, forehead, mouth.

    NOT realistic, but it tests whether models latch onto skin-texture cues.
    """
    name = "aging_wrinkles"

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        lms = _safe_landmarks(ctx)
        if lms is None:
            return image.copy()

        out = image.copy()
        h, w = image.shape[:2]
        eye_l = lms[MP_INDICES["left_eye_outer"]]
        eye_r = lms[MP_INDICES["right_eye_outer"]]
        eye_w = float(np.linalg.norm(eye_r - eye_l))
        if eye_w < 5:
            return out

        # Crow's feet: short diagonal lines outside each outer eye.
        for cx, cy, side in [(eye_l[0], eye_l[1], -1), (eye_r[0], eye_r[1], 1)]:
            base_x, base_y = int(cx + side * eye_w * 0.15), int(cy)
            for k, dy in enumerate([-6, 0, 6]):
                length = int(eye_w * 0.18)
                pt1 = (base_x, base_y + dy)
                pt2 = (base_x + side * length, base_y + dy + (-2 if dy < 0 else 2))
                cv2.line(out, pt1, pt2, (60, 50, 50), 1, cv2.LINE_AA)

        # Forehead horizontal lines.
        forehead = lms[MP_INDICES["forehead_top"]]
        brow_l = lms[MP_INDICES["left_brow"]]
        brow_r = lms[MP_INDICES["right_brow"]]
        fy_top = int(forehead[1] + (brow_l[1] - forehead[1]) * 0.3)
        fy_bot = int(forehead[1] + (brow_l[1] - forehead[1]) * 0.7)
        x0 = int(brow_l[0] + 5)
        x1 = int(brow_r[0] - 5)
        for fy in (fy_top, fy_bot):
            # Wavy line.
            xs = np.linspace(x0, x1, 40)
            ys = fy + np.sin(np.linspace(0, 4 * math.pi, 40)) * 1.2
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(out, [pts], False, (70, 60, 60), 1, cv2.LINE_AA)

        # Nasolabial folds: diagonal lines from nose corners to mouth corners.
        nose_bot = lms[MP_INDICES["nose_bottom"]]
        mouth_l = lms[61]
        mouth_r = lms[291]
        cv2.line(out,
                 (int(nose_bot[0] - eye_w * 0.15), int(nose_bot[1])),
                 (int(mouth_l[0]), int(mouth_l[1] - eye_w * 0.05)),
                 (75, 60, 60), 1, cv2.LINE_AA)
        cv2.line(out,
                 (int(nose_bot[0] + eye_w * 0.15), int(nose_bot[1])),
                 (int(mouth_r[0]), int(mouth_r[1] - eye_w * 0.05)),
                 (75, 60, 60), 1, cv2.LINE_AA)
        return out
