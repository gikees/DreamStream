"""Main receiver pipeline: chains interpolator -> upscaler -> enhancer."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from dreamstream.config import ReceiverConfig
from dreamstream.receiver.enhancer import Enhancer, PassthroughEnhancer, SVDEnhancer
from dreamstream.receiver.interpolator import (
    DuplicationInterpolator,
    Interpolator,
    OpticalFlowInterpolator,
    RIFEInterpolator,
)
from dreamstream.receiver.upscaler import BicubicUpscaler, RealESRGANUpscaler, Upscaler

logger = logging.getLogger(__name__)


class ReconstructionPipeline:
    """Orchestrates frame interpolation, upscaling, and enhancement."""

    def __init__(
        self,
        config: ReceiverConfig,
        device: torch.device,
        output_size: Tuple[int, int] = (1280, 720),
    ) -> None:
        self._config = config
        self._device = device
        self._output_size = output_size  # (width, height)
        self._prev_frame: np.ndarray | None = None

        # Model status tracking
        self.model_status: Dict[str, str] = {}

        # Build components with graceful fallback
        self._interpolator = self._build_interpolator(config)
        self._reliable_upscaler = self._build_reliable_upscaler(config)
        self._dream_upscaler = self._build_dream_upscaler(config)
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

    def _build_reliable_upscaler(self, config: ReceiverConfig) -> Upscaler:
        self.model_status["reliable_upscaler"] = "bicubic"
        logger.info("Reliable upscaler: bicubic")
        return BicubicUpscaler()

    def _build_dream_upscaler(self, config: ReceiverConfig) -> Upscaler:
        weights_path = config.weights_dir / "RealESRGAN_x4.pth"
        try:
            upscaler = RealESRGANUpscaler(weights_path, self._device)
            self.model_status["dream_upscaler"] = "real_esrgan_x4"
            return upscaler
        except Exception as e:
            logger.warning("Real-ESRGAN unavailable (%s), falling back to bicubic", e)
            self.model_status["dream_upscaler"] = "bicubic"
            return BicubicUpscaler()

    def _build_enhancer(self, config: ReceiverConfig) -> Enhancer:
        try:
            enhancer = SVDEnhancer(self._device)
            self.model_status["enhancer"] = "svd"
            logger.info("Enhancer: SVD (Stable Video Diffusion)")
            return enhancer
        except Exception as e:
            logger.info("SVD unavailable (%s), falling back to passthrough", e)
            self.model_status["enhancer"] = "passthrough"
            logger.info("Enhancer: passthrough (baseline)")
            return PassthroughEnhancer()

    @property
    def num_output_frames(self) -> int:
        """Number of output frames to generate per input frame."""
        return max(1, round(self._config.output_fps / 3.0))

    def process_frame(
        self, frame: np.ndarray, edges: np.ndarray | None = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Interpolate once, then fork into reliable and dream paths.

        Returns (reliable_frames, dream_frames). Fixes the prev_frame bug
        where separate calls would see stale/updated state inconsistently.
        """
        interpolated = self._interpolator.interpolate(
            frame, self._prev_frame, self.num_output_frames
        )
        self._prev_frame = frame

        reliable = [
            self._reliable_upscaler.upscale(f, self._output_size) for f in interpolated
        ]

        upscaled_dream = [
            self._dream_upscaler.upscale(f, self._output_size) for f in interpolated
        ]
        dream = self._enhancer.enhance_batch(upscaled_dream, edges=edges)

        return reliable, dream
