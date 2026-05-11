"""Stable Diffusion inpainting manipulations.

These are higher-fidelity than classical overlays, but require a GPU and a
model download (~5 GB). Each manipulation specifies:
  * the region to inpaint (built from face landmarks),
  * a positive prompt,
  * a negative prompt.

The same SD pipeline is shared across all GenAI manipulations to avoid reloading.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from manipulations.base import FaceContext, Manipulation, register
from manipulations.landmarks import MP_INDICES
from utils import get_logger

LOG = get_logger(__name__)


@dataclass
class _SDConfig:
    model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    strength: float = 0.85


_sd_config = _SDConfig()


def configure_sd(model_id: str | None = None, guidance_scale: float | None = None,
                 num_inference_steps: int | None = None, strength: float | None = None) -> None:
    """Override defaults from a config file."""
    if model_id is not None:
        _sd_config.model_id = model_id
    if guidance_scale is not None:
        _sd_config.guidance_scale = guidance_scale
    if num_inference_steps is not None:
        _sd_config.num_inference_steps = num_inference_steps
    if strength is not None:
        _sd_config.strength = strength


@lru_cache(maxsize=1)
def _get_pipeline():
    """Lazy-load the Stable Diffusion inpainting pipeline. Cached as a singleton."""
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as e:
        raise ImportError(
            "diffusers / torch are required for GenAI manipulations. "
            "Install with: pip install -r requirements-genai.txt"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    LOG.info("Loading SD inpainting pipeline %s on %s (%s)", _sd_config.model_id, device, dtype)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _sd_config.model_id, torch_dtype=dtype, safety_checker=None,
    ).to(device)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_attention_slicing()
    return pipe


def _run_inpaint(image_bgr: np.ndarray, mask: np.ndarray,
                 prompt: str, negative_prompt: str) -> np.ndarray:
    """Run SD inpainting and return a BGR uint8 image.

    The pipeline operates at 512×512 internally; we resize before & after.
    ``mask`` is uint8, with 255 indicating the region to be filled.
    """
    from PIL import Image
    import torch

    pipe = _get_pipeline()
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb).resize((512, 512), Image.LANCZOS)
    pil_mask = Image.fromarray(mask).resize((512, 512), Image.NEAREST)

    generator = torch.Generator(device=pipe.device).manual_seed(0)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_img,
        mask_image=pil_mask,
        guidance_scale=_sd_config.guidance_scale,
        num_inference_steps=_sd_config.num_inference_steps,
        strength=_sd_config.strength,
        generator=generator,
    ).images[0]

    out = result.resize((w, h), Image.LANCZOS)
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    return out_bgr


def _mustache_mask(image_shape: tuple[int, int], lms: np.ndarray) -> np.ndarray:
    """Build a mask covering the philtrum / mustache area."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    nose_bot = lms[MP_INDICES["nose_bottom"]]
    upper_lip = lms[MP_INDICES["upper_lip_top"]].mean(axis=0)
    mouth_l = lms[61]
    mouth_r = lms[291]

    cx = (mouth_l[0] + mouth_r[0]) / 2
    band_w = float(np.linalg.norm(mouth_r - mouth_l)) * 1.4
    band_h = float(np.linalg.norm(nose_bot - upper_lip)) * 2.2
    cy = (nose_bot[1] + upper_lip[1]) / 2

    cv2.ellipse(mask, (int(cx), int(cy)), (int(band_w / 2), int(band_h / 2)),
                0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return mask


def _beard_mask(image_shape: tuple[int, int], lms: np.ndarray) -> np.ndarray:
    """Build a mask covering the chin / lower jaw."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                   152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    poly = lms[jaw_indices].astype(np.int32)
    mouth_l = lms[61].astype(np.int32)
    mouth_r = lms[291].astype(np.int32)
    full = np.vstack([poly, mouth_r[None, :], mouth_l[None, :]])
    cv2.fillPoly(mask, [full], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 7)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return mask


def _eyes_mask(image_shape: tuple[int, int], lms: np.ndarray) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for ring in (
        [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
        [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    ):
        pts = lms[ring].astype(np.float32)
        center = pts.mean(axis=0)
        expanded = ((pts - center) * 1.7 + center + np.array([0, -3])).astype(np.int32)
        cv2.fillPoly(mask, [expanded], 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return mask


# ============================================================
# Manipulation classes
# ============================================================

class _GenAIManipulation(Manipulation):
    """Common scaffolding for inpainting-based manipulations."""
    prompt: str = ""
    negative_prompt: str = "blurry, low quality, distorted, deformed, watermark"

    def _build_mask(self, image_shape: tuple[int, int], lms: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, image: np.ndarray, ctx: FaceContext) -> np.ndarray:
        if ctx.landmarks is None:
            return image.copy()
        mask = self._build_mask(image.shape[:2], ctx.landmarks)
        if mask.sum() == 0:
            return image.copy()
        return _run_inpaint(image, mask, self.prompt, self.negative_prompt)


@register
class PromptMustache(_GenAIManipulation):
    name = "prompt_mustache"
    prompt = "a person with a thick dark realistic mustache, photo, sharp focus"
    negative_prompt = "blurry, cartoon, child, painting, watermark, low quality"

    def _build_mask(self, image_shape, lms):
        return _mustache_mask(image_shape, lms)


@register
class PromptBeard(_GenAIManipulation):
    name = "prompt_beard"
    prompt = "a person with a full dark beard covering the chin, photo, sharp focus"
    negative_prompt = "blurry, cartoon, child, painting, watermark, low quality"

    def _build_mask(self, image_shape, lms):
        return _beard_mask(image_shape, lms)


@register
class PromptMakeup(_GenAIManipulation):
    name = "prompt_makeup"
    prompt = "heavy dramatic eye makeup, eye shadow and eyeliner, photo, sharp focus"
    negative_prompt = "blurry, cartoon, painting, watermark, low quality"

    def _build_mask(self, image_shape, lms):
        return _eyes_mask(image_shape, lms)
