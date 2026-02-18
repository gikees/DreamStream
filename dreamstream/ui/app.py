"""Gradio Blocks UI — single gr.Video showing side-by-side comparison."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from dreamstream.config import (
    PipelineConfig,
    ReceiverConfig,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _process_video(video_path: str) -> tuple[str | None, dict | None]:
    """Run the enhancement pipeline on an uploaded video."""
    if not video_path:
        return None, None

    from dreamstream.cli import _run_pipeline

    tmp_dir = Path(tempfile.mkdtemp(prefix="dreamstream_"))

    cfg = PipelineConfig(
        receiver=ReceiverConfig(),
        device=resolve_device(),
        out_dir=tmp_dir,
    )

    metrics = _run_pipeline(cfg, Path(video_path))
    comparison_path = tmp_dir / "comparison.mp4"

    if comparison_path.exists():
        return str(comparison_path), metrics
    return None, metrics


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="DreamStream") as app:
        gr.Markdown("# DreamStream\nAI-powered video enhancement")

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Upload Video")
                process_btn = gr.Button("Enhance", variant="primary")
            with gr.Column(scale=2):
                output_video = gr.Video(label="Input vs Enhanced")

        metrics_json = gr.JSON(label="Pipeline Metrics")

        process_btn.click(
            fn=_process_video,
            inputs=[input_video],
            outputs=[output_video, metrics_json],
        )

    return app
