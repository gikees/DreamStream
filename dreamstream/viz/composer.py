"""2x2 grid stitcher with labels for visualization output."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compose_grid(
    degraded: np.ndarray,
    reliable: np.ndarray,
    dream: np.ndarray,
    heatmap: np.ndarray,
    cell_size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Compose a 2x2 grid from four video frames with labels.

    Layout:
        [Degraded (240p@3fps)] [Reliable]
        [Dream]                [Uncertainty]

    Args:
        degraded: BGR frame (low-res input).
        reliable: BGR frame (upscaled, no enhancement).
        dream: BGR frame (upscaled + enhanced).
        heatmap: BGR heatmap frame.
        cell_size: (width, height) per cell. Default (640,360) → 1280x720 grid.

    Returns:
        BGR frame of shape (cell_size[1]*2, cell_size[0]*2, 3).
    """
    cw, ch = cell_size

    # Resize each to cell size with appropriate interpolation
    deg_cell = cv2.resize(degraded, (cw, ch), interpolation=cv2.INTER_NEAREST)
    rel_cell = cv2.resize(reliable, (cw, ch), interpolation=cv2.INTER_CUBIC)
    drm_cell = cv2.resize(dream, (cw, ch), interpolation=cv2.INTER_CUBIC)
    hm_cell = cv2.resize(heatmap, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # Draw labels
    labels = [
        (deg_cell, "Degraded (240p@3fps)"),
        (rel_cell, "Reliable"),
        (drm_cell, "Dream"),
        (hm_cell, "Uncertainty"),
    ]

    for cell, label in labels:
        _draw_label(cell, label)

    # Stitch: top row, bottom row, then vstack
    top = np.hstack([deg_cell, rel_cell])
    bottom = np.hstack([drm_cell, hm_cell])
    grid = np.vstack([top, bottom])

    return grid


def _draw_label(frame: np.ndarray, text: str) -> None:
    """Draw a white-on-black label in the top-left corner of a frame (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (0, 0), (tw + 10, th + baseline + 10), bg_color, -1)
    # Text
    cv2.putText(frame, text, (5, th + 5), font, scale, color, thickness, cv2.LINE_AA)
