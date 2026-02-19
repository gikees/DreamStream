# DreamStream – Claude Code Guidelines

## Project Overview
AI-powered video enhancement pipeline optimized for NVIDIA GB10 (Grace Blackwell).
Takes low-quality, low-framerate, or low-resolution video and enhances it into high-fidelity, high-framerate footage using a local AI pipeline (Interpolation -> Upscaling -> Generative Detailing).

## Hard Constraints
- **NO xformers, flash-attn, or custom CUDA kernels** — compiling C++ on ARM during a hackathon is forbidden.
- **NO cloud API calls** — all inference must run locally.
- Dependencies: torch, torchvision, opencv-python, diffusers, gradio. Nothing else for core functionality.
- Python 3.10+.

## Architecture Rules
- Modular structure: stages/, viz/, metrics/, ui/, models/. Keep modules independent.
- The pipeline runs sequentially: Interpolate -> Upscale -> Enhance.
- Single enhancement path (no reliable/dream split) — use best available models.
- The pipeline must ALWAYS work without optional weights (graceful degradation).
- Optional models (RIFE, Real-ESRGAN, LCM) are loaded behind try/except. Missing weights = fallback to baseline.
- `weights/` directory is gitignored. Never commit model weights.
- Vendored model code lives in `dreamstream/models/<model_name>/`. Must be pure PyTorch — no custom CUDA ops.
- When vendoring a model, **always verify the architecture against the actual checkpoint** (`state_dict` key names and shapes). Different RIFE versions share the same file name but have incompatible architectures.

## GB10 / Unified Memory
- Use `torch.device('cuda')` when available. Log `torch.cuda.get_device_name()` at startup.
- Minimize CPU<->GPU tensor transfers. Convert numpy->tensor ONCE, process on device, convert back ONCE.
- Use `non_blocking=True` for `.to(device)` calls.
- Do NOT use pin_memory tricks — on unified memory it's a no-op.

## Video I/O
- `dreamstream/video_io.py`: `VideoMeta`, `probe_video()`, `read_frames()` for input handling.
- Codec fallback order: avc1 -> mp4v -> XVID. Use `create_video_writer()` from config.py.
- FFmpeg stderr is redirected to /dev/null during codec probing to suppress `h264_v4l2m2m` noise on GB10.
- All frame dimensions must be even (width and height) for codec compatibility.

## Gradio UI
- ONE single gr.Video component showing a pre-stitched 1x2 side-by-side comparison.
- Never use multiple separate gr.Video components (they desync during playback).

## Conventions
- Use dataclasses for configuration (config.py).
- ABCs for pipeline components (Interpolator, Upscaler, Enhancer) with baseline implementations.
- Logging via `logging` module, not print().
- BGR color space throughout (OpenCV native). Only convert at model boundaries.
- Do NOT add `Co-Authored-By` or any co-authorship lines to commit messages.

## Remote Machine (GB10)
- SSH: `sshpass -p '123456' ssh -o StrictHostKeyChecking=no dell@100.89.249.36`
- Project path: `~/DreamStream`, venv at `.venv/` (activate with `source .venv/bin/activate`)
- GPU: NVIDIA GB10 (Grace Blackwell), CUDA available
- To deploy: push to origin, then `git pull && pip install -e .` on remote
- `gdown` installed at `/home/dell/.local/bin/gdown` (not on PATH — use full path or venv)

## Pipeline Fallback Chains
Each pipeline component tries AI models first, then falls back to baselines:
- **Interpolation**: RIFE → Optical Flow (Farneback) → Frame Duplication
- **Upscaling**: Real-ESRGAN x4 → Bicubic
- **Enhancement**: SD img2img + ControlNet Canny → SD img2img → Passthrough

## Weight Management
- Weight paths: `weights/RealESRGAN_x4.pth`, `weights/rife/flownet.pkl`
- Direct URLs in `WEIGHT_URLS` dict, Google Drive IDs in `GDRIVE_WEIGHTS` dict (both in `cli.py`).
- Download via `python -m dreamstream download-weights` (requires `gdown` for RIFE weights).
- `_download_weights()` creates subdirectories automatically.
- RIFE uses official Practical-RIFE v4.26 weights (Google Drive). Do NOT use HuggingFace mirrors — they serve incompatible older architectures.

## CLI
Entry point: `python -m dreamstream` (or `dreamstream` if pip-installed).
- `run -i <video> [-o out_dir] [--output-height 720] [--output-fps 24] [-v]`
- `ui [--share] [-v]`
- `download-weights [--weights-dir weights/] [-v]`

## Pipeline Output
- 2 videos: `enhanced.mp4`, `comparison.mp4` (1x2 side-by-side: Input vs Enhanced)
- All re-encoded to H.264 via ffmpeg (`remux_to_h264()`) for broad player compatibility.
- `metrics.json` with latency, FPS, model status, and processing stats.

## Build Order
Phase 1: Scaffold -> Phase 2: AI Pipeline (Interpolation + Upscale) -> Phase 3: Viz -> Phase 4: UI -> Phase 5: Generative Enhancers
