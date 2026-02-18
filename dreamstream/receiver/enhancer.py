"""Frame enhancement: optional AI-based refinement of upscaled frames."""

from __future__ import annotations

import abc
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class Enhancer(abc.ABC):
    """Abstract base for frame enhancers."""

    @abc.abstractmethod
    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        """Enhance an upscaled frame, optionally using edge hints."""

    def enhance_batch(
        self, frames: List[np.ndarray], edges: np.ndarray | None = None
    ) -> List[np.ndarray]:
        """Enhance a batch of frames. Default: per-frame fallback."""
        return [self.enhance(f, edges=edges) for f in frames]


class PassthroughEnhancer(Enhancer):
    """Baseline: no enhancement (Dream view = Reliable view)."""

    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        return frame


class SVDEnhancer(Enhancer):
    """Temporal enhancement via Stable Video Diffusion.

    Conditions on the first frame of each group and generates temporally
    coherent output, blended with the upscaled input to preserve structure.
    """

    def __init__(self, device: "torch.device", alpha: float = 0.35) -> None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self._device = device
        self._alpha = alpha

        self._pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        self._pipe.to(device)
        logger.info("SVD enhancer loaded (device=%s, dtype=%s)", device, dtype)

    def enhance(self, frame: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        return frame

    def enhance_batch(
        self, frames: List[np.ndarray], edges: np.ndarray | None = None
    ) -> List[np.ndarray]:
        import cv2
        import torch
        from PIL import Image

        if not frames:
            return frames

        n = len(frames)

        # Condition on first frame (convert BGR -> RGB -> PIL)
        cond_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        cond_pil = Image.fromarray(cond_rgb)

        # SVD generates 14 frames by default; we sample n evenly
        num_svd_frames = 14
        with torch.inference_mode():
            output = self._pipe(
                cond_pil,
                num_frames=num_svd_frames,
                num_inference_steps=8,
                decode_chunk_size=4,
            )
        svd_frames = output.frames[0]  # list of PIL images

        # Sample n frames evenly from SVD output
        indices = [int(i * (num_svd_frames - 1) / max(n - 1, 1)) for i in range(n)]
        sampled = [svd_frames[idx] for idx in indices]

        # Blend SVD output with upscaled input
        results: List[np.ndarray] = []
        for inp_frame, svd_pil in zip(frames, sampled):
            svd_rgb = np.array(svd_pil)
            svd_bgr = cv2.cvtColor(svd_rgb, cv2.COLOR_RGB2BGR)
            # Resize SVD output to match input frame dimensions
            h, w = inp_frame.shape[:2]
            svd_bgr = cv2.resize(svd_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            # Blend: alpha * SVD + (1-alpha) * input
            blended = cv2.addWeighted(svd_bgr, self._alpha, inp_frame, 1.0 - self._alpha, 0)
            results.append(blended)

        return results
