"""Enhancement pipeline: chains interpolator -> upscaler -> enhancer."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from dreamstream.config import ReceiverConfig
from dreamstream.receiver.enhancer import Enhancer, PassthroughEnhancer
from dreamstream.receiver.interpolator import (
    DuplicationInterpolator,
    Interpolator,
    OpticalFlowInterpolator,
    RIFEInterpolator,
)
from dreamstream.receiver.upscaler import BicubicUpscaler, RealESRGANUpscaler, Upscaler

logger = logging.getLogger(__name__)


class EnhancementPipeline:
    """Orchestrates frame interpolation, upscaling, and enhancement."""

    def __init__(
        self,
        config: ReceiverConfig,
        device: torch.device,
        input_fps: float,
        output_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        self._config = config
        self._device = device
        self._input_fps = input_fps
        self._output_size = output_size  # (width, height)
        self._prev_frame: np.ndarray | None = None

        # Model status tracking
        self.model_status: Dict[str, str] = {}

        # Build components with graceful fallback
        self._interpolator = self._build_interpolator(config)
        self._upscaler = self._build_upscaler(config)
        self._enhancer = self._build_enhancer(config)

    def _build_interpolator(self, config: ReceiverConfig) -> Interpolator:
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

    def _build_upscaler(self, config: ReceiverConfig) -> Upscaler:
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

    def _build_enhancer(self, config: ReceiverConfig) -> Enhancer:
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
            enhanced = self._enhancer.enhance(up)
            results.append(enhanced)

        return results
