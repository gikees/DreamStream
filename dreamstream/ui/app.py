"""Gradio Blocks UI — single gr.Video showing pre-stitched 2x2 grid."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from dreamstream.config import (
    PipelineConfig,
    Profile,
    ReceiverConfig,
    SenderConfig,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _process_video(video_path: str, profile: str) -> tuple[str | None, dict | None]:
    """Run the pipeline on an uploaded video, return (grid_path, metrics_dict)."""
    if not video_path:
        return None, None

    from dreamstream.cli import _run_pipeline

    tmp_dir = Path(tempfile.mkdtemp(prefix="dreamstream_"))

    cfg = PipelineConfig(
        sender=SenderConfig(),
        receiver=ReceiverConfig(),
        profile=Profile(profile),
        device=resolve_device(),
        out_dir=tmp_dir,
    )

    metrics = _run_pipeline(cfg, Path(video_path))
    grid_path = tmp_dir / "stitched_grid.mp4"

    if grid_path.exists():
        return str(grid_path), metrics
    return None, metrics


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="DreamStream") as app:
        gr.Markdown("# DreamStream\nBandwidth-resilient generative video reconstruction")

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label="Upload Video")
                profile_dropdown = gr.Dropdown(
                    choices=[p.value for p in Profile],
                    value=Profile.LOW_RGB.value,
                    label="Sender Profile",
                )
                process_btn = gr.Button("Process", variant="primary")
            with gr.Column(scale=2):
                output_video = gr.Video(label="Reconstruction Grid (2x2)")

        metrics_json = gr.JSON(label="Pipeline Metrics")

        gr.Markdown(
            "*Dream view is AI-generated. "
            "Heatmap shows uncertainty — warm regions are more likely hallucinated.*"
        )

        process_btn.click(
            fn=_process_video,
            inputs=[input_video, profile_dropdown],
            outputs=[output_video, metrics_json],
        )

    return app
