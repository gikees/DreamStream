# DreamStream – Claude Code Guidelines

## Project Overview
Bandwidth-resilient generative video reconstruction pipeline for NVIDIA GB10 (Grace Blackwell).
Sender degrades video to 240p@3fps hints; receiver reconstructs 720p@24fps using local inference only.

## Hard Constraints
- **NO xformers, flash-attn, or custom CUDA kernels** — compiling C++ on ARM during a hackathon is forbidden.
- **NO cloud API calls** — all inference must run locally.
- Dependencies: torch, torchvision, opencv-python, diffusers, gradio. Nothing else for core functionality.
- Python 3.10+.

## Architecture Rules
- Modular structure: sender_sim/, receiver/, viz/, metrics/, ui/, models/. Keep modules independent.
- The receiver must ALWAYS work without optional weights (graceful degradation).
- Optional models (RIFE, Real-ESRGAN, LCM) are loaded behind try/except. Missing weights = fallback to baseline.
- `weights/` directory is gitignored. Never commit model weights.
- Vendored model code lives in `dreamstream/models/<model_name>/`. Must be pure PyTorch — no custom CUDA ops.

## GB10 / Unified Memory
- Use `torch.device('cuda')` when available. Log `torch.cuda.get_device_name()` at startup.
- Minimize CPU<->GPU tensor transfers. Convert numpy->tensor ONCE, process on device, convert back ONCE.
- Use `non_blocking=True` for `.to(device)` calls.
- Do NOT use pin_memory tricks — on unified memory it's a no-op.

## Video I/O
- Codec fallback order: avc1 -> mp4v -> XVID. Use `create_video_writer()` from config.py.
- All frame dimensions must be even (width and height) for codec compatibility.
- Use INTER_AREA for downscaling, INTER_CUBIC for upscaling.
- Process frames via generators — never load entire video into memory.

## Gradio UI
- ONE single gr.Video component showing a pre-stitched 2x2 OpenCV grid.
- Never use four separate gr.Video components (they desync during playback).

## Conventions
- Use dataclasses for configuration (config.py).
- ABCs for receiver components (Interpolator, Upscaler, Enhancer) with baseline implementations.
- Logging via `logging` module, not print().
- BGR color space throughout (OpenCV native). Only convert at model boundaries.
- Do NOT add `Co-Authored-By` or any co-authorship lines to commit messages.

## Remote Machine (GB10)
- SSH: `sshpass -p '123456' ssh -o StrictHostKeyChecking=no dell@100.89.249.36`

## Receiver Fallback Chains
Each receiver component tries AI models first, then falls back to baselines:
- **Interpolation**: RIFE → Optical Flow (Farneback) → Frame Duplication
- **Upscaling (dream path)**: Real-ESRGAN x4 → Bicubic
- **Upscaling (reliable path)**: Always Bicubic
- **Enhancement**: Passthrough (LCM placeholder for future)

## Weight Management
- Weight paths: `weights/RealESRGAN_x4.pth`, `weights/rife/flownet.pkl`
- URLs registered in `WEIGHT_URLS` dict in `cli.py`. Download via `python -m dreamstream download-weights`.
- `_download_weights()` creates subdirectories automatically. New weights: add entry to `WEIGHT_URLS`.

## Profiles
- `low_rgb`: 240p RGB @ 3fps (default)
- `low_rgb_edges`: 240p RGB @ 3fps + Canny edge hints (low=50, high=150)

## CLI
Entry point: `python -m dreamstream` (or `dreamstream` if pip-installed).
- `run -i <video> [-p profile] [-o out_dir] [--output-height 720] [--output-fps 24] [-v]`
- `ui [--share] [-v]`
- `download-weights [--weights-dir weights/] [-v]`

## Pipeline Output
- 5 videos: `degraded.mp4`, `reliable.mp4`, `dream.mp4`, `heatmap.mp4`, `stitched_grid.mp4`
- All re-encoded to H.264 via ffmpeg (`remux_to_h264()`) for broad player compatibility.
- `metrics.json` with latency, bandwidth, model status, and processing stats.

## Build Order
Phase 1: Scaffold -> Phase 2: Sender -> Phase 3: Receiver -> Phase 4: Viz -> Phase 5: UI -> Phase 6: Optional models
