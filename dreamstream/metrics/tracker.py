"""Pipeline metrics: FPS, latency, model status tracking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Final metrics from a pipeline run."""

    source_resolution: str
    output_resolution: str
    source_fps: float
    output_fps: float
    total_input_frames: int
    total_output_frames: int
    avg_latency_ms: float
    p95_latency_ms: float
    models_loaded: Dict[str, str] = field(default_factory=dict)
    processing_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_resolution": self.source_resolution,
            "output_resolution": self.output_resolution,
            "source_fps": self.source_fps,
            "output_fps": self.output_fps,
            "total_input_frames": self.total_input_frames,
            "total_output_frames": self.total_output_frames,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
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

    def __init__(self, output_fps: float, source_fps: float) -> None:
        self._output_fps = output_fps
        self._source_fps = source_fps
        self._frame_latencies: List[float] = []
        self._frame_count = 0
        self._output_count = 0
        self._start_time = 0.0
        self._frame_start = 0.0
        self._source_resolution = ""

    def start(self, source_resolution: str = "") -> None:
        self._start_time = time.monotonic()
        self._source_resolution = source_resolution

    def record_frame(self, output_frames: List[np.ndarray]) -> None:
        """Record metrics for one input frame and its output frames."""
        now = time.monotonic()
        if self._frame_count > 0:
            self._frame_latencies.append((now - self._frame_start) * 1000)
        self._frame_start = now

        self._frame_count += 1
        self._output_count += len(output_frames)

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

        return PipelineMetrics(
            source_resolution=self._source_resolution or "(unknown)",
            output_resolution="(see output files)",
            source_fps=self._source_fps,
            output_fps=self._output_fps,
            total_input_frames=self._frame_count,
            total_output_frames=self._output_count,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            models_loaded=model_status,
            processing_time_s=elapsed,
        )
