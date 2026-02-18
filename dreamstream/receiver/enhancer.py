"""Frame enhancement: optional AI-based refinement of upscaled frames."""

from __future__ import annotations

import abc
import logging

import numpy as np

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
