"""Frame enhancement: optional AI-based refinement of upscaled frames."""

from __future__ import annotations

import abc
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Enhancer(abc.ABC):
    """Abstract base for frame enhancers."""

    @abc.abstractmethod
    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        """Enhance an upscaled frame, optionally using edge hints."""


class PassthroughEnhancer(Enhancer):
    """Baseline: no enhancement (Dream view = Reliable view)."""

    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        return frame


class SDImg2ImgEnhancer(Enhancer):
    """Stable Diffusion img2img enhancer using diffusers."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        strength: float = 0.3,
        num_steps: int = 10,
        prompt: str = "high quality, sharp, detailed",
        negative_prompt: str = "blurry, noisy, artifacts, low quality",
        guidance_scale: float = 7.5,
    ) -> None:
        from diffusers import AutoPipelineForImage2Image

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None,
            requires_safety_checker=False,
        )
        self._pipe.to(device)
        self._pipe.enable_attention_slicing()

        self._device = device
        self._strength = strength
        self._num_steps = num_steps
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._guidance_scale = guidance_scale
        self._generator = torch.Generator(device=device).manual_seed(42)

        logger.info(
            "SD img2img enhancer loaded: model=%s, strength=%.2f, steps=%d",
            model_id, strength, num_steps,
        )

    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        from PIL import Image

        orig_h, orig_w = frame.shape[:2]

        # BGR -> RGB -> PIL
        rgb = frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        # Resize to 512px on the long side, dims divisible by 8
        if orig_w >= orig_h:
            new_w = 512
            new_h = int(512 * orig_h / orig_w)
        else:
            new_h = 512
            new_w = int(512 * orig_w / orig_h)
        new_w = max(8, new_w - new_w % 8)
        new_h = max(8, new_h - new_h % 8)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Run img2img
        result = self._pipe(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            image=pil_img,
            strength=self._strength,
            num_inference_steps=self._num_steps,
            guidance_scale=self._guidance_scale,
            generator=self._generator,
        )
        out_pil = result.images[0]

        # Resize back to original dimensions and convert RGB -> BGR
        out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)
        out_rgb = np.array(out_pil)
        out_bgr = out_rgb[:, :, ::-1].copy()

        return out_bgr
