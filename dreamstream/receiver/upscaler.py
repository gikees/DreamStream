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
    def upscale(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Upscale frame to target_size (width, height)."""


class BicubicUpscaler(Upscaler):
    """Baseline: bicubic interpolation via OpenCV."""

    def upscale(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
