"""Frame interpolation: fill temporal gaps between low-fps source frames."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Interpolator(abc.ABC):
    """Abstract base for frame interpolators."""

    @abc.abstractmethod
    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        """Produce num_output_frames from frame (and optionally prev_frame)."""


class DuplicationInterpolator(Interpolator):
    """Baseline: duplicate the current frame N times."""

    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        # Return references — callers must .copy() before mutating
        return [frame] * num_output_frames


class OpticalFlowInterpolator(Interpolator):
    """CPU-only optical-flow interpolation using Farneback.

    Blends prev_frame→frame across num_output_frames using flow warping.
    Falls back to duplication if prev_frame is None.
    """

    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        if prev_frame is None:
            return [frame] * num_output_frames

        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        h, w = frame.shape[:2]
        results: List[np.ndarray] = []

        for i in range(num_output_frames):
            alpha = (i + 1) / num_output_frames
            # Warp prev_frame toward frame by alpha fraction of the flow
            flow_map = flow * alpha
            map_x = np.arange(w, dtype=np.float32)[None, :] + flow_map[..., 0]
            map_y = np.arange(h, dtype=np.float32)[:, None] + flow_map[..., 1]
            warped = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
            # Blend warped previous with current
            blended = cv2.addWeighted(warped, 1.0 - alpha, frame, alpha, 0)
            results.append(blended)

        return results


class RIFEInterpolator(Interpolator):
    """AI frame interpolation via ECCV2022-RIFE IFNet (pure PyTorch).

    Uses recursive midpoint interpolation to generate num_output_frames
    between prev_frame and frame. Falls back to duplication when prev_frame
    is None.
    """

    def __init__(self, weights_path: Path | str, device: torch.device) -> None:
        from dreamstream.models.rife.ifnet import Model

        self._device = device
        self._model = Model()
        self._model.load_model(str(weights_path), -1)
        self._model.eval()
        logger.info("RIFE loaded from %s", weights_path)

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """BGR uint8 numpy → (1, 3, H, W) float32 tensor on device."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        return t.unsqueeze(0).to(self._device, non_blocking=True)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) float32 tensor → BGR uint8 numpy."""
        rgb = (tensor.squeeze(0).clamp(0, 1) * 255).byte()
        rgb_np = rgb.cpu().permute(1, 2, 0).numpy()
        return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    def _pad_to_multiple(self, img: torch.Tensor, multiple: int = 32):
        """Pad tensor height/width to nearest multiple (RIFE requirement)."""
        _, _, h, w = img.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph or pw:
            img = torch.nn.functional.pad(img, (0, pw, 0, ph), mode="replicate")
        return img, h, w

    @torch.inference_mode()
    def interpolate(
        self, frame: np.ndarray, prev_frame: np.ndarray | None, num_output_frames: int
    ) -> List[np.ndarray]:
        if prev_frame is None:
            return [frame] * num_output_frames

        img0 = self._to_tensor(prev_frame)
        img1 = self._to_tensor(frame)

        # Pad to multiple of 32 for RIFE
        img0_p, orig_h, orig_w = self._pad_to_multiple(img0)
        img1_p, _, _ = self._pad_to_multiple(img1)

        # Generate evenly-spaced timestamps excluding 0 (prev) including 1 (current)
        timestamps = [(i + 1) / num_output_frames for i in range(num_output_frames)]
        results: List[np.ndarray] = []

        for t in timestamps:
            if abs(t - 1.0) < 1e-6:
                results.append(frame)
            else:
                mid = self._model.inference(img0_p, img1_p, timestep=t)
                mid = mid[:, :, :orig_h, :orig_w]
                results.append(self._to_numpy(mid))

        return results
