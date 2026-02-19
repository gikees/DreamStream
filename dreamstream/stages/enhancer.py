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
    """Stable Diffusion img2img enhancer using diffusers, optionally with ControlNet Canny."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        strength: float = 0.3,
        num_steps: int = 10,
        prompt: str = "high quality, sharp, detailed",
        negative_prompt: str = "blurry, noisy, artifacts, low quality",
        guidance_scale: float = 7.5,
        controlnet_model_id: str | None = None,
    ) -> None:
        import cv2 as _cv2  # noqa: F811 – ensure cv2 available at init time

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._has_controlnet = False

        # Try loading ControlNet first
        if controlnet_model_id is not None:
            try:
                from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

                controlnet = ControlNetModel.from_pretrained(
                    controlnet_model_id, torch_dtype=dtype,
                )
                self._pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    model_id, controlnet=controlnet, torch_dtype=dtype,
                    safety_checker=None, requires_safety_checker=False,
                )
                self._has_controlnet = True
                logger.info("ControlNet Canny loaded: %s", controlnet_model_id)
            except Exception as e:
                logger.warning("ControlNet unavailable (%s), falling back to plain img2img", e)

        # Fall back to plain img2img if ControlNet didn't load
        if not self._has_controlnet:
            from diffusers import AutoPipelineForImage2Image

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
            "SD img2img enhancer loaded: model=%s, strength=%.2f, steps=%d, controlnet=%s",
            model_id, strength, num_steps, self._has_controlnet,
        )

    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        import cv2
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

        # Build pipeline kwargs
        pipe_kwargs = dict(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            image=pil_img,
            strength=self._strength,
            num_inference_steps=self._num_steps,
            guidance_scale=self._guidance_scale,
            generator=self._generator,
        )

        # Add ControlNet Canny conditioning if available
        if self._has_controlnet:
            if edges is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
            # Resize edges to match diffusion input and convert to 3-channel PIL
            edges_resized = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canny_pil = Image.fromarray(edges_resized).convert("RGB")
            pipe_kwargs["control_image"] = canny_pil

        result = self._pipe(**pipe_kwargs)
        out_pil = result.images[0]

        # Resize back to original dimensions and convert RGB -> BGR
        out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)
        out_rgb = np.array(out_pil)
        out_bgr = out_rgb[:, :, ::-1].copy()

        return out_bgr
