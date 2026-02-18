# DreamStream

AI-powered video enhancement pipeline for NVIDIA GB10 (Grace Blackwell).

Takes low-quality, low-framerate, or low-resolution video and enhances it into high-fidelity, high-framerate footage using local AI models (RIFE interpolation, Real-ESRGAN upscaling).

## Architecture

```
Input Video → [Enhancement Pipeline] → Enhanced Output
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
 Interpolate       Upscale        Enhance
 (RIFE/flow)    (ESRGAN/bicubic)  (optional)
     │               │               │
     └───────┬───────┘               │
             ├───────────────────────┘
             ▼
   enhanced.mp4 + comparison.mp4
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download optional AI model weights
python -m dreamstream download-weights

# Enhance a video via CLI
python -m dreamstream run -i input.mp4 -v

# Launch the Gradio web UI
python -m dreamstream ui
```

## Output Files

The pipeline produces 3 files in the output directory:

| File | Description |
|------|-------------|
| `enhanced.mp4` | AI-enhanced output (interpolated + upscaled) |
| `comparison.mp4` | Side-by-side: Input (naive upscale) vs Enhanced (AI) |
| `metrics.json` | Pipeline metrics (latency, FPS, models used) |

## CLI Reference

```bash
# Full options
python -m dreamstream run --help

# Custom output resolution and FPS
python -m dreamstream run -i input.mp4 --output-height 480 --output-fps 15

# Launch UI with public link
python -m dreamstream ui --share
```

## Optional Models

Download weights with `python -m dreamstream download-weights`, or place them manually:

| Model | Path | Effect |
|-------|------|--------|
| RIFE v4.26 | `weights/rife/flownet.pkl` | AI temporal interpolation (replaces optical flow) |
| Real-ESRGAN x4 | `weights/RealESRGAN_x4.pth` | AI super-resolution (replaces bicubic) |

All models are optional — the pipeline gracefully degrades to baseline algorithms when weights are missing.

## Fallback Chains

- **Interpolation**: RIFE → Optical Flow (Farneback) → Frame Duplication
- **Upscaling**: Real-ESRGAN x4 → Bicubic
- **Enhancement**: Passthrough (placeholder for future ControlNet/LCM)
