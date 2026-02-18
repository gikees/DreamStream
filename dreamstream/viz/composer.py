"""Side-by-side comparison composer for visualization output."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compose_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    cell_size: Tuple[int, int] = (640, 720),
) -> np.ndarray:
    """Compose a 1x2 side-by-side comparison frame.

    Layout:
        [Input] [Enhanced]

    Args:
        original: BGR frame (naive upscale of input).
        enhanced: BGR frame (AI-enhanced output).
        cell_size: (width, height) per cell. Default (640,720) -> 1280x720 total.

    Returns:
        BGR frame of shape (cell_size[1], cell_size[0]*2, 3).
    """
    cw, ch = cell_size

    orig_cell = cv2.resize(original, (cw, ch), interpolation=cv2.INTER_CUBIC)
    enh_cell = cv2.resize(enhanced, (cw, ch), interpolation=cv2.INTER_CUBIC)

    _draw_label(orig_cell, "Input")
    _draw_label(enh_cell, "Enhanced")

    return np.hstack([orig_cell, enh_cell])


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
