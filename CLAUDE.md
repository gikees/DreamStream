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
- Modular structure: sender_sim/, receiver/, viz/, metrics/, ui/. Keep modules independent.
- The receiver must ALWAYS work without optional weights (graceful degradation).
- Optional models (RIFE, Real-ESRGAN, LCM) are loaded behind try/except. Missing weights = fallback to baseline.
- `weights/` directory is gitignored. Never commit model weights.

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

## Build Order
Phase 1: Scaffold -> Phase 2: Sender -> Phase 3: Receiver -> Phase 4: Viz -> Phase 5: UI -> Phase 6: Optional models
