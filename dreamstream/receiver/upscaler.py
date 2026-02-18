"""Spatial upscaling: enlarge low-resolution frames to output resolution."""

from __future__ import annotations

import abc
import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Upscaler(abc.ABC):
    """Abstract base for spatial upscalers."""

    @abc.abstractmethod
    def upscale(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        """Upscale frame to target_height, maintaining aspect ratio."""


class BicubicUpscaler(Upscaler):
    """Baseline: bicubic interpolation via OpenCV."""

    def upscale(self, frame: np.ndarray, target_height: int) -> np.ndarray:
        h, w = frame.shape[:2]
        aspect = w / h
        target_w = int(target_height * aspect)
        # Enforce even dimensions for codec compatibility
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_height if target_height % 2 == 0 else target_height + 1
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
