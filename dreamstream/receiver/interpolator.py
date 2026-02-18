"""Frame interpolation: fill temporal gaps between low-fps source frames."""

from __future__ import annotations

import abc
import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Interpolator(abc.ABC):
    """Abstract base for frame interpolators."""

    @abc.abstractmethod
    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        """Produce num_output_frames from frame (and optionally prev_frame)."""


class DuplicationInterpolator(Interpolator):
    """Baseline: duplicate the current frame N times."""

    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        # Return references — callers must .copy() before mutating
        return [frame] * num_output_frames


class OpticalFlowInterpolator(Interpolator):
    """CPU-only optical-flow interpolation using Farneback.

    Blends prev_frame→frame across num_output_frames using flow warping.
    Falls back to duplication if prev_frame is None.
    """

    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        if prev_frame is None:
            return [frame] * num_output_frames

        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        h, w = frame.shape[:2]
        results: List[np.ndarray] = []

        for i in range(num_output_frames):
            alpha = (i + 1) / num_output_frames
            # Warp prev_frame toward frame by alpha fraction of the flow
            flow_map = flow * alpha
            map_x = np.arange(w, dtype=np.float32)[None, :] + flow_map[..., 0]
            map_y = np.arange(h, dtype=np.float32)[:, None] + flow_map[..., 1]
            warped = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
            # Blend warped previous with current
            blended = cv2.addWeighted(warped, 1.0 - alpha, frame, alpha, 0)
            results.append(blended)

        return results
