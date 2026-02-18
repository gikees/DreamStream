"""Main receiver pipeline: chains interpolator -> upscaler -> enhancer."""

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
)
from dreamstream.receiver.upscaler import BicubicUpscaler, Upscaler

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
        self._upscaler = self._build_upscaler(config)
        self._enhancer = self._build_enhancer(config)

    def _build_interpolator(self, config: ReceiverConfig) -> Interpolator:
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
        return max(1, round(self._config.output_fps / 3.0))

    def get_reliable_frames(
        self, frame: np.ndarray, edges: np.ndarray | None = None
    ) -> List[np.ndarray]:
        """Interpolate + upscale (no enhancement)."""
        interpolated = self._interpolator.interpolate(
            frame, self._prev_frame, self.num_output_frames
        )
        upscaled = [
            self._upscaler.upscale(f, self._output_size) for f in interpolated
        ]
        self._prev_frame = frame
        return upscaled

    def get_dream_frames(
        self, frame: np.ndarray, edges: np.ndarray | None = None
    ) -> List[np.ndarray]:
        """Interpolate + upscale + enhance."""
        interpolated = self._interpolator.interpolate(
            frame, self._prev_frame, self.num_output_frames
        )
        results = []
        for f in interpolated:
            up = self._upscaler.upscale(f, self._output_size)
            enhanced = self._enhancer.enhance(up, edges=edges)
            results.append(enhanced)
        return results
