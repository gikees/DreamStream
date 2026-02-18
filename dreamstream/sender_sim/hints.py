"""Edge-detection hints for bandwidth-efficient structure encoding."""

from __future__ import annotations

import cv2
import numpy as np


def compute_canny_edges(
    frame: np.ndarray, low: int = 50, high: int = 150
) -> np.ndarray:
    """Compute Canny edges from a BGR frame.

    Returns a single-channel uint8 edge map (0 or 255).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def edges_to_rgb(edge_map: np.ndarray) -> np.ndarray:
    """Convert single-channel edge map to BGR for visualization."""
    return cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
