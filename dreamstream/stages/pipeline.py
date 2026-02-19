"""Enhancement pipeline: chains interpolator -> upscaler -> enhancer."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from dreamstream.config import StagesConfig
from dreamstream.stages.enhancer import Enhancer, PassthroughEnhancer, SDImg2ImgEnhancer
from dreamstream.stages.interpolator import (
    DuplicationInterpolator,
    Interpolator,
    OpticalFlowInterpolator,
    RIFEInterpolator,
)
from dreamstream.stages.upscaler import BicubicUpscaler, RealESRGANUpscaler, Upscaler

logger = logging.getLogger(__name__)


class EnhancementPipeline:
    """Orchestrates frame interpolation, upscaling, and enhancement."""

    def __init__(
        self,
        config: StagesConfig,
        device: torch.device,
        input_fps: float,
        output_size: Tuple[int, int] = (1280, 720),
        input_size: Tuple[int, int] | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._input_fps = input_fps
        self._output_size = output_size  # (width, height)
        self._prev_frame: np.ndarray | None = None

        # Skip AI upscaler when input already meets or exceeds target
        self._needs_ai_upscale = True
        if input_size is not None:
            in_w, in_h = input_size
            out_w, out_h = output_size
            if in_h >= out_h and in_w >= out_w:
                self._needs_ai_upscale = False
                logger.info(
                    "Input %dx%d >= target %dx%d — skipping AI upscaler",
                    in_w, in_h, out_w, out_h,
                )

        # Model status tracking
        self.model_status: Dict[str, str] = {}

        # Build components with graceful fallback
        self._interpolator = self._build_interpolator(config)
        self._upscaler = self._build_upscaler(config)
        self._enhancer = self._build_enhancer(config)

    def _build_interpolator(self, config: StagesConfig) -> Interpolator:
        # Try RIFE first (AI interpolation)
        weights_path = config.weights_dir / "rife" / "flownet.pkl"
        try:
            interp = RIFEInterpolator(weights_path, self._device)
            self.model_status["interpolator"] = "rife"
            logger.info("Interpolator: RIFE (AI)")
            return interp
        except Exception as e:
            logger.info("RIFE unavailable (%s), trying optical flow", e)

        # Fall back to optical flow
        try:
            interp = OpticalFlowInterpolator()
            self.model_status["interpolator"] = "optical_flow"
            logger.info("Interpolator: optical flow (Farneback)")
            return interp
        except Exception:
            self.model_status["interpolator"] = "frame_duplication"
            logger.info("Interpolator: frame duplication (baseline)")
            return DuplicationInterpolator()

    def _build_upscaler(self, config: StagesConfig) -> Upscaler:
        if not self._needs_ai_upscale:
            self.model_status["upscaler"] = "bicubic (input >= target)"
            logger.info("Upscaler: bicubic (input already meets target resolution)")
            return BicubicUpscaler()

        weights_path = config.weights_dir / "RealESRGAN_x4.pth"
        try:
            upscaler = RealESRGANUpscaler(weights_path, self._device)
            self.model_status["upscaler"] = "real_esrgan_x4"
            logger.info("Upscaler: Real-ESRGAN x4 (AI)")
            return upscaler
        except Exception as e:
            logger.warning("Real-ESRGAN unavailable (%s), falling back to bicubic", e)
            self.model_status["upscaler"] = "bicubic"
            logger.info("Upscaler: bicubic (baseline)")
            return BicubicUpscaler()

    def _build_enhancer(self, config: StagesConfig) -> Enhancer:
        try:
            enhancer = SDImg2ImgEnhancer(
                model_id=config.enhancer_model_id,
                device=self._device,
                strength=config.enhancer_strength,
                num_steps=config.enhancer_steps,
                prompt=config.enhancer_prompt,
                negative_prompt=config.enhancer_negative_prompt,
                guidance_scale=config.enhancer_guidance_scale,
                controlnet_model_id=config.controlnet_model_id,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
            )
            if enhancer._has_controlnet:
                self.model_status["enhancer"] = "sd_img2img_controlnet"
                logger.info("Enhancer: SD img2img + ControlNet Canny (AI)")
            else:
                self.model_status["enhancer"] = "sd_img2img"
                logger.info("Enhancer: SD img2img (AI)")
            return enhancer
        except Exception as e:
            logger.warning("SD img2img unavailable (%s), falling back to passthrough", e)
            self.model_status["enhancer"] = "passthrough"
            logger.info("Enhancer: passthrough (baseline)")
            return PassthroughEnhancer()

    @property
    def num_output_frames(self) -> int:
        """Number of output frames to generate per input frame."""
        if self._input_fps < self._config.output_fps:
            return max(1, round(self._config.output_fps / self._input_fps))
        return 1

    def process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """Interpolate, upscale, and enhance a single input frame.

        Returns a list of enhanced output frames.
        """
        interpolated = self._interpolator.interpolate(
            frame, self._prev_frame, self.num_output_frames
        )
        self._prev_frame = frame

        results = []
        for f in interpolated:
            up = self._upscaler.upscale(f, self._output_size)
            edges = cv2.Canny(
                cv2.GaussianBlur(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                50, 150,
            )
            enhanced = self._enhancer.enhance(up, edges=edges)
            results.append(enhanced)

        return results
