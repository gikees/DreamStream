"""Central configuration: dataclasses, device setup, video writer utility."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StagesConfig:
    output_height: int = 720
    output_fps: float | None = None  # None = match input FPS
    weights_dir: Path = field(default_factory=lambda: Path("weights"))
    # SD img2img enhancer settings
    enhancer_model_id: str = "runwayml/stable-diffusion-v1-5"
    enhancer_strength: float = 0.3
    enhancer_steps: int = 10
    enhancer_prompt: str = "high quality, sharp, detailed"
    enhancer_negative_prompt: str = "blurry, noisy, artifacts, low quality"
    enhancer_guidance_scale: float = 7.5
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"


@dataclass
class PipelineConfig:
    stages: StagesConfig = field(default_factory=StagesConfig)
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

    # Suppress all FFmpeg stderr spam during codec probing — the fallback is intentional
    import os
    prev_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        for codec in FOURCC_CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
            if writer.isOpened():
                logger.debug("VideoWriter opened with codec %s for %s", codec, path)
                return writer
            writer.release()
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
        cv2.utils.logging.setLogLevel(prev_log_level)

    raise RuntimeError(
        f"No working codec found for {path}. Tried: {FOURCC_CANDIDATES}"
    )


# ---------------------------------------------------------------------------
# Post-processing: re-encode to H.264 if ffmpeg is available
# ---------------------------------------------------------------------------

def remux_to_h264(video_path: Path | str) -> None:
    """Re-encode a video file to H.264 using ffmpeg (in-place).

    If ffmpeg is not installed, logs a warning and leaves the file as-is.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found — skipping H.264 re-encode for %s", video_path)
        return

    tmp_path = video_path.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        "-an",  # no audio
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        tmp_path.replace(video_path)
        logger.info("Re-encoded %s to H.264", video_path.name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("H.264 re-encode failed for %s: %s", video_path.name, exc)
        tmp_path.unlink(missing_ok=True)
