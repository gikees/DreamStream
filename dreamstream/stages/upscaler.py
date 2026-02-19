"""Spatial upscaling: enlarge low-resolution frames to output resolution."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
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


class RealESRGANUpscaler(Upscaler):
    """AI super-resolution via ai-forever/Real-ESRGAN (pure PyTorch, no C++)."""

    def __init__(self, weights_path: Path | str, device, scale: int = 4) -> None:
        from PIL import Image  # noqa: F401 – verify PIL available
        from RealESRGAN import RealESRGAN

        self._model = RealESRGAN(device, scale=scale)
        self._model.load_weights(str(weights_path))
        self._scale = scale
        logger.info("Real-ESRGAN loaded (scale=%d) from %s", scale, weights_path)

    def upscale(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        from PIL import Image

        # BGR → RGB → PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        sr_img = self._model.predict(pil_img)  # returns PIL Image at scale×
        # PIL → numpy BGR
        sr_bgr = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2BGR)
        # Resize to exact target if 4× doesn't match
        if (sr_bgr.shape[1], sr_bgr.shape[0]) != target_size:
            sr_bgr = cv2.resize(sr_bgr, target_size, interpolation=cv2.INTER_AREA)
        return sr_bgr
