"""CLI entry point for DreamStream."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dreamstream.config import (
    PipelineConfig,
    Profile,
    ReceiverConfig,
    SenderConfig,
    remux_to_h264,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dreamstream",
        description="DreamStream — bandwidth-resilient generative video reconstruction",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Process a video through the pipeline")
    run_p.add_argument("-i", "--input", required=True, type=Path, help="Input video path")
    run_p.add_argument(
        "-p",
        "--profile",
        default="low_rgb",
        choices=[p.value for p in Profile],
        help="Sender profile (default: low_rgb)",
    )
    run_p.add_argument("-o", "--out-dir", default=Path("outputs"), type=Path, help="Output directory")
    run_p.add_argument("--output-height", default=720, type=int, help="Receiver output height (default: 720)")
    run_p.add_argument("--output-fps", default=24.0, type=float, help="Receiver output FPS (default: 24)")
    run_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # --- ui ---
    ui_p = sub.add_parser("ui", help="Launch Gradio web UI")
    ui_p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ui_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    # --- download-weights ---
    dl_p = sub.add_parser("download-weights", help="Download optional AI model weights")
    dl_p.add_argument(
        "--weights-dir", default=Path("weights"), type=Path, help="Weights directory"
    )
    dl_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return parser


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_pipeline(cfg: PipelineConfig, input_path: Path) -> dict:
    """Execute the full sender→receiver→viz pipeline.

    Returns a metrics dict.
    """
    from dreamstream.metrics.tracker import MetricsTracker
    from dreamstream.receiver.pipeline import ReconstructionPipeline
    from dreamstream.sender_sim.degrader import degrade_video, probe_video
    from dreamstream.sender_sim.hints import compute_canny_edges, edges_to_rgb
    from dreamstream.viz.composer import compose_grid
    from dreamstream.viz.heatmap import generate_heatmap
    from dreamstream.config import create_video_writer

    # Probe source video
    meta = probe_video(input_path)
    logger.info(
        "Source: %dx%d @ %.1f fps, %d frames",
        meta.width, meta.height, meta.fps, meta.frame_count,
    )

    # Compute output dimensions (maintain aspect ratio)
    aspect = meta.width / meta.height
    out_h = cfg.receiver.output_height
    out_w = int(out_h * aspect)
    out_w = out_w if out_w % 2 == 0 else out_w + 1

    # Build receiver pipeline with explicit output size
    pipeline = ReconstructionPipeline(cfg.receiver, cfg.device, output_size=(out_w, out_h))
    num_output = round(cfg.receiver.output_fps / cfg.sender.target_fps)

    # Metrics tracker
    tracker = MetricsTracker(cfg)

    # Degraded dimensions
    deg_h = cfg.sender.target_height
    deg_w = int(deg_h * aspect)
    deg_w = deg_w if deg_w % 2 == 0 else deg_w + 1

    # Grid dimensions: 2x2 of 640x360 cells = 1280x720
    grid_w, grid_h = 1280, 720

    # Create video writers
    degraded_writer = create_video_writer(
        cfg.out_dir / "degraded.mp4", cfg.sender.target_fps, (deg_w, deg_h)
    )
    reliable_writer = create_video_writer(
        cfg.out_dir / "reliable.mp4", cfg.receiver.output_fps, (out_w, out_h)
    )
    dream_writer = create_video_writer(
        cfg.out_dir / "dream.mp4", cfg.receiver.output_fps, (out_w, out_h)
    )
    heatmap_writer = create_video_writer(
        cfg.out_dir / "heatmap.mp4", cfg.receiver.output_fps, (out_w, out_h)
    )
    grid_writer = create_video_writer(
        cfg.out_dir / "stitched_grid.mp4", cfg.receiver.output_fps, (grid_w, grid_h)
    )

    tracker.start(source_resolution=f"{meta.width}x{meta.height}")
    frame_idx = 0

    for deg_frame, orig_idx in degrade_video(input_path, cfg.sender):
        edges = None
        if cfg.profile == Profile.LOW_RGB_EDGES:
            edges = compute_canny_edges(
                deg_frame, cfg.sender.canny_low, cfg.sender.canny_high
            )

        reliable_frames, dream_frames = pipeline.process_frame(deg_frame, edges)

        # Write degraded (1 frame per source frame at sender fps)
        degraded_writer.write(deg_frame)

        # Write reliable + dream + heatmap + grid (num_output frames per source frame)
        for i, (rel_f, drm_f) in enumerate(zip(reliable_frames, dream_frames)):
            reliable_writer.write(rel_f)
            dream_writer.write(drm_f)

            # Heatmap based on dream frame
            hmap = generate_heatmap(drm_f, edges=edges)
            heatmap_writer.write(hmap)

            # Compose grid
            grid = compose_grid(deg_frame.copy(), rel_f.copy(), drm_f.copy(), hmap.copy())
            grid_writer.write(grid)

        tracker.record_frame(deg_frame, reliable_frames, dream_frames)
        frame_idx += 1

    # Release all writers
    for w in [degraded_writer, reliable_writer, dream_writer, heatmap_writer, grid_writer]:
        w.release()

    # Re-encode to H.264 for broad player compatibility
    for name in ["degraded.mp4", "reliable.mp4", "dream.mp4", "heatmap.mp4", "stitched_grid.mp4"]:
        remux_to_h264(cfg.out_dir / name)

    metrics = tracker.finalize(pipeline.model_status)
    metrics_path = cfg.out_dir / "metrics.json"
    metrics.to_json(metrics_path)
    logger.info("Metrics written to %s", metrics_path)
    logger.info("Processed %d degraded frames → %d output frames", frame_idx, frame_idx * num_output)

    return metrics.to_dict()


WEIGHT_URLS = {
    "RealESRGAN_x4.pth": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth",
}

# Google Drive file IDs for weights that aren't on HuggingFace
GDRIVE_WEIGHTS = {
    "rife/flownet.pkl": "1gViYvvQrtETBgU1w8axZSsr7YUuw31uy",  # Practical-RIFE v4.26
}


def _download_weights(weights_dir: Path) -> None:
    """Download optional AI model weights to weights_dir."""
    import urllib.request
    import zipfile
    import tempfile

    weights_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in WEIGHT_URLS.items():
        dest = weights_dir / filename
        if dest.exists():
            logger.info("Already exists: %s", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", url, dest)
        urllib.request.urlretrieve(url, dest)
        logger.info("Downloaded %s (%.1f MB)", filename, dest.stat().st_size / 1e6)

    # Google Drive weights (requires gdown)
    for filename, file_id in GDRIVE_WEIGHTS.items():
        dest = weights_dir / filename
        if dest.exists():
            logger.info("Already exists: %s", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError:
            logger.warning(
                "gdown not installed — cannot download %s from Google Drive. "
                "Install with: pip install gdown", filename,
            )
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "model.zip"
            logger.info("Downloading %s from Google Drive (id=%s)", filename, file_id)
            gdown.download(id=file_id, output=str(zip_path), quiet=False)
            with zipfile.ZipFile(zip_path) as zf:
                # Find flownet.pkl inside the zip
                pkl_names = [n for n in zf.namelist() if n.endswith("flownet.pkl")]
                if not pkl_names:
                    logger.error("No flownet.pkl found in downloaded zip for %s", filename)
                    continue
                with zf.open(pkl_names[0]) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            logger.info("Downloaded %s (%.1f MB)", filename, dest.stat().st_size / 1e6)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _setup_logging(getattr(args, "verbose", False))

    if args.command == "run":
        input_path = args.input
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

        cfg = PipelineConfig(
            sender=SenderConfig(),
            receiver=ReceiverConfig(
                output_height=args.output_height,
                output_fps=args.output_fps,
            ),
            profile=Profile(args.profile),
            device=resolve_device(),
            out_dir=args.out_dir,
        )
        _run_pipeline(cfg, input_path)

    elif args.command == "download-weights":
        _download_weights(args.weights_dir)

    elif args.command == "ui":
        from dreamstream.ui.app import create_app

        app = create_app()
        app.launch(share=args.share)
