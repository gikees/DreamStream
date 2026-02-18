"""Video I/O: metadata probing and frame reading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """Metadata extracted from a source video."""

    width: int
    height: int
    fps: float
    frame_count: int


def probe_video(path: Path | str) -> VideoMeta:
    """Read video metadata without decoding frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def read_frames(path: Path | str) -> Iterator[np.ndarray]:
    """Generator yielding BGR frames at native FPS."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
