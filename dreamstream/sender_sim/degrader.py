"""Degrade source video to ultra-low-bandwidth frames (240p @ 3fps)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from dreamstream.config import SenderConfig

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


def _compute_target_size(
    src_width: int, src_height: int, target_height: int
) -> Tuple[int, int]:
    """Compute target (width, height) maintaining aspect ratio with even dimensions."""
    aspect = src_width / src_height
    w = int(target_height * aspect)
    # Enforce even dimensions for codec compatibility
    w = w if w % 2 == 0 else w + 1
    h = target_height if target_height % 2 == 0 else target_height + 1
    return w, h


def degrade_video(
    input_path: Path | str, config: SenderConfig
) -> Iterator[Tuple[np.ndarray, int]]:
    """Generator yielding (degraded_frame, original_frame_index).

    Subsamples temporally to target_fps and spatially to target_height.
    Uses read-and-discard for skipped frames (faster than seeking in compressed video).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip_interval = max(1, round(src_fps / config.target_fps))
    target_w, target_h = _compute_target_size(src_w, src_h, config.target_height)

    logger.info(
        "Degrading: %dx%d@%.1ffps → %dx%d@%.1ffps (skip every %d frames)",
        src_w, src_h, src_fps, target_w, target_h, config.target_fps, skip_interval,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_interval == 0:
                resized = cv2.resize(
                    frame, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                yield resized, frame_idx

            frame_idx += 1
    finally:
        cap.release()

    logger.info("Degraded %d frames, yielded %d", frame_idx, frame_idx // skip_interval)
