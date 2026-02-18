"""Pipeline metrics: kbps, FPS, latency, model status tracking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from dreamstream.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Final metrics from a pipeline run."""

    profile: str
    source_resolution: str
    degraded_resolution: str
    output_resolution: str
    sender_fps: float
    output_fps: float
    total_degraded_frames: int
    total_output_frames: int
    avg_latency_ms: float
    p95_latency_ms: float
    estimated_raw_kbps: float
    estimated_compressed_kbps: float
    models_loaded: Dict[str, str] = field(default_factory=dict)
    processing_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "source_resolution": self.source_resolution,
            "degraded_resolution": self.degraded_resolution,
            "output_resolution": self.output_resolution,
            "sender_fps": self.sender_fps,
            "output_fps": self.output_fps,
            "total_degraded_frames": self.total_degraded_frames,
            "total_output_frames": self.total_output_frames,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "estimated_raw_kbps": round(self.estimated_raw_kbps, 1),
            "estimated_compressed_kbps": round(self.estimated_compressed_kbps, 1),
            "models_loaded": self.models_loaded,
            "processing_time_s": round(self.processing_time_s, 2),
        }

    def to_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricsTracker:
    """Tracks per-frame metrics during pipeline execution."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._frame_latencies: List[float] = []
        self._degraded_sizes: List[int] = []
        self._degraded_frames: List[np.ndarray] = []
        self._frame_count = 0
        self._output_count = 0
        self._start_time = 0.0
        self._frame_start = 0.0

    def start(self) -> None:
        self._start_time = time.monotonic()

    def record_frame(
        self,
        degraded: np.ndarray,
        reliable_frames: List[np.ndarray],
        dream_frames: List[np.ndarray],
    ) -> None:
        """Record metrics for one degraded frame and its output frames."""
        now = time.monotonic()
        if self._frame_count > 0:
            self._frame_latencies.append((now - self._frame_start) * 1000)
        self._frame_start = now

        self._frame_count += 1
        self._output_count += len(reliable_frames)

        # Estimate compressed size via JPEG encoding at Q=50
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        _, buf = cv2.imencode(".jpg", degraded, encode_params)
        self._degraded_sizes.append(len(buf))

    def finalize(self, model_status: Dict[str, str]) -> PipelineMetrics:
        """Compute final metrics."""
        elapsed = time.monotonic() - self._start_time

        # Latency stats
        if self._frame_latencies:
            avg_lat = sum(self._frame_latencies) / len(self._frame_latencies)
            sorted_lat = sorted(self._frame_latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        else:
            avg_lat = 0.0
            p95_lat = 0.0

        # Bandwidth estimation
        sender_cfg = self._config.sender
        deg_h = sender_cfg.target_height
        # Estimate width from typical 16:9 aspect
        deg_w = int(deg_h * 16 / 9)
        deg_w = deg_w if deg_w % 2 == 0 else deg_w + 1

        # Raw kbps: H * W * 3 channels * 8 bits * fps / 1000
        raw_kbps = deg_h * deg_w * 3 * 8 * sender_cfg.target_fps / 1000

        # Compressed kbps from measured JPEG sizes
        if self._degraded_sizes:
            avg_bytes = sum(self._degraded_sizes) / len(self._degraded_sizes)
            compressed_kbps = avg_bytes * 8 * sender_cfg.target_fps / 1000
        else:
            compressed_kbps = 0.0

        # Output resolution from receiver config
        out_h = self._config.receiver.output_height
        out_w = int(out_h * 16 / 9)
        out_w = out_w if out_w % 2 == 0 else out_w + 1

        return PipelineMetrics(
            profile=self._config.profile.value,
            source_resolution="(from input)",
            degraded_resolution=f"{deg_w}x{deg_h}",
            output_resolution=f"{out_w}x{out_h}",
            sender_fps=sender_cfg.target_fps,
            output_fps=self._config.receiver.output_fps,
            total_degraded_frames=self._frame_count,
            total_output_frames=self._output_count,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            estimated_raw_kbps=raw_kbps,
            estimated_compressed_kbps=compressed_kbps,
            models_loaded=model_status,
            processing_time_s=elapsed,
        )
