# DreamStream

Bandwidth-resilient generative video reconstruction pipeline for NVIDIA GB10 (Grace Blackwell).

The **sender** degrades video to ultra-low-bandwidth structure hints (240p @ 3fps), and the **receiver** reconstructs smooth, interpretable video (720p @ 24fps) using only local inference.

## Architecture

```
Source Video → [Sender Sim] → 240p@3fps hints → [Receiver Pipeline] → 720p@24fps outputs
                                                        │
                              ┌──────────────────────────┼──────────────────────┐
                              ▼                          ▼                      ▼
                        Interpolate               Upscale (bicubic)      Enhance (optional)
                     (frame dup/flow)                                    (diffusion/SR)
                              │                          │                      │
                              └──────────┬───────────────┘                      │
                                         ▼                                      ▼
                                   Reliable View                          Dream View
                                         │                                      │
                                         ├──────────────────────────────────────┘
                                         ▼
                              [Heatmap + 2x2 Grid Composer] → stitched_grid.mp4
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a video via CLI
python -m dreamstream run -i input.mp4 -p low_rgb -v

# Launch the Gradio web UI
python -m dreamstream ui
```

## Profiles

| Profile | Sender Output | Description |
|---------|--------------|-------------|
| `low_rgb` | 240p RGB @ 3fps | Spatially + temporally degraded color frames |
| `low_rgb_edges` | 240p RGB @ 3fps + Canny edges | Adds edge structure hints for uncertainty mapping |

## Output Files

The pipeline produces 6 files in the output directory:

| File | Resolution | FPS | Description |
|------|-----------|-----|-------------|
| `degraded.mp4` | 240p | 3 | Sender output (what gets transmitted) |
| `reliable.mp4` | 720p | 24 | Interpolated + upscaled (no AI enhancement) |
| `dream.mp4` | 720p | 24 | Interpolated + upscaled + enhanced (AI-generated) |
| `heatmap.mp4` | 720p | 24 | Uncertainty visualization (warm = more hallucinated) |
| `stitched_grid.mp4` | 1280x720 | 24 | 2x2 grid of all four views with labels |
| `metrics.json` | — | — | Pipeline metrics (kbps, latency, models used) |

## Reliable vs Dream

- **Reliable view**: Only uses deterministic operations (optical flow interpolation, bicubic upscaling). What you see is a faithful reconstruction of the transmitted data.
- **Dream view**: Applies optional AI enhancement (super-resolution, diffusion refinement). Looks better but may hallucinate details not present in the source.
- **Heatmap**: Shows where the dream view is most uncertain — pixels far from detected edges are more likely to be hallucinated.

## CLI Reference

```bash
# Full options
python -m dreamstream run --help

# Process with edge hints
python -m dreamstream run -i input.mp4 -p low_rgb_edges -o my_outputs/ -v

# Custom output resolution and FPS
python -m dreamstream run -i input.mp4 --output-height 480 --output-fps 15

# Launch UI with public link
python -m dreamstream ui --share
```

## Optional Models

Place model weights in the `weights/` directory for enhanced reconstruction:

| Model | Directory | Effect |
|-------|----------|--------|
| RIFE | `weights/rife/` | Temporal interpolation (replaces optical flow) |
| Real-ESRGAN | `weights/realesrgan/` | 4x super-resolution (replaces bicubic) |
| LCM Diffusion | `weights/lcm/` | Generative enhancement (replaces passthrough) |

All models are optional — the pipeline gracefully degrades to baseline algorithms when weights are missing.
