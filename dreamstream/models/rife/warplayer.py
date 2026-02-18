"""Backward warp using F.grid_sample (pure PyTorch).

Vendored from Practical-RIFE (https://github.com/hzwer/Practical-RIFE).
Adapted: uses tensor device instead of a global torch.device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Backward-warp tenInput according to tenFlow.

    Args:
        tenInput: (B, C, H, W) image tensor.
        tenFlow: (B, 2, H, W) optical flow tensor (x, y displacements).

    Returns:
        Warped tensor of same shape as tenInput.
    """
    B, _, H, W = tenFlow.shape
    tenHorizontal = (
        torch.linspace(-1.0, 1.0, W, device=tenFlow.device)
        .view(1, 1, 1, W)
        .expand(B, -1, H, -1)
    )
    tenVertical = (
        torch.linspace(-1.0, 1.0, H, device=tenFlow.device)
        .view(1, 1, H, 1)
        .expand(B, -1, -1, W)
    )
    tenGrid = torch.cat([tenHorizontal, tenVertical], dim=1)
    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        dim=1,
    )
    return F.grid_sample(
        input=tenInput,
        grid=(tenGrid + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
