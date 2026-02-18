"""Central configuration: profiles, dataclasses, device setup, video writer utility."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class Profile(enum.Enum):
    LOW_RGB = "low_rgb"
    LOW_RGB_EDGES = "low_rgb_edges"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SenderConfig:
    target_height: int = 240
    target_fps: float = 3.0
    canny_low: int = 50
    canny_high: int = 150


@dataclass
class ReceiverConfig:
    output_height: int = 720
    output_fps: float = 24.0
    weights_dir: Path = field(default_factory=lambda: Path("weights"))


@dataclass
class PipelineConfig:
    sender: SenderConfig = field(default_factory=SenderConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    profile: Profile = Profile.LOW_RGB
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    out_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device() -> torch.device:
    """Return CUDA device if available, else CPU. Logs GPU name for confirmation."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        logger.info("Using GPU: %s", name)
    else:
        dev = torch.device("cpu")
        logger.info("CUDA not available — using CPU")
    return dev


# ---------------------------------------------------------------------------
# Video writer with codec fallback
# ---------------------------------------------------------------------------

FOURCC_CANDIDATES = ["avc1", "mp4v", "XVID"]


def create_video_writer(
    path: Path | str,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """Create a VideoWriter, trying codecs in fallback order.

    Args:
        path: Output file path.
        fps: Target frames per second.
        frame_size: (width, height) of output frames.

    Returns:
        An opened cv2.VideoWriter.

    Raises:
        RuntimeError: If no codec works.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for codec in FOURCC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if writer.isOpened():
            logger.debug("VideoWriter opened with codec %s for %s", codec, path)
            return writer
        writer.release()

    raise RuntimeError(
        f"No working codec found for {path}. Tried: {FOURCC_CANDIDATES}"
    )
