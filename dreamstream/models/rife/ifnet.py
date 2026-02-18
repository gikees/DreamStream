"""IFNet v4.6 — Practical-RIFE intermediate flow network (inference only).

Vendored from https://github.com/hzwer/Practical-RIFE.
Architecture matched to the public flownet.pkl weights (3 student blocks,
uniform c=90, separate flow/mask deconv heads).
Adapted: device-agnostic, no xformers/flash-attn, teacher block ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamstream.models.rife.warplayer import warp

logger = logging.getLogger(__name__)


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=True,
        ),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    """Single scale block with separate flow/mask deconv heads."""

    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock0(x) + x
        x = self.convblock1(x) + x
        x = self.convblock2(x) + x
        x = self.convblock3(x) + x
        flow = self.conv1(x)
        mask = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False)
        return flow, mask


class IFNet(nn.Module):
    """Multi-scale IFNet: 3 student blocks + teacher (unused at inference)."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=90)
        self.block1 = IFBlock(7 + 4, c=90)
        self.block2 = IFBlock(7 + 4, c=90)
        self.block_tea = IFBlock(10 + 4, c=90)

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[int] | None = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = [4, 2, 1]

        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # Timestep map: same spatial dims as input, filled with timestep value
        timestep_tensor = (x[:, :1].clone() * 0 + 1) * timestep

        # Flow and mask start at zeros — accumulated across blocks (coarse-to-fine)
        B, _, H, W = x.shape
        flow = torch.zeros(B, 4, H, W, device=x.device)
        mask = torch.zeros(B, 1, H, W, device=x.device)

        blocks = [self.block0, self.block1, self.block2]
        for block, scale in zip(blocks, scale_list):
            flow_d, mask_d = block(
                torch.cat((img0, img1, timestep_tensor, flow), dim=1),
                scale=scale,
            )
            flow = flow + flow_d
            mask = mask + mask_d

        # Final warp and blend using learned mask
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)


class Model:
    """Wrapper matching the API expected by RIFEInterpolator.

    Methods:
        load_model(path, rank) — load flownet.pkl weights
        eval() — set to eval mode
        inference(img0, img1, timestep) — interpolate single frame
    """

    def __init__(self) -> None:
        self.flownet = IFNet()
        self.device = torch.device("cpu")

    def load_model(self, path: str, rank: int = 0) -> None:
        """Load weights from flownet.pkl.

        Args:
            path: Directory containing flownet.pkl, or direct path to the .pkl file.
            rank: Unused (kept for API compat). Pass -1 for CPU.
        """
        p = Path(path)
        if p.is_dir():
            p = p / "flownet.pkl"

        self.device = torch.device("cpu") if rank < 0 else torch.device("cuda")

        state = torch.load(p, map_location="cpu", weights_only=True)
        # Handle keys with or without "module." prefix (DDP artifact)
        cleaned = {}
        for k, v in state.items():
            cleaned[k.replace("module.", "")] = v
        self.flownet.load_state_dict(cleaned, strict=False)
        self.flownet.to(self.device)
        logger.info("RIFE flownet loaded from %s on %s", p, self.device)

    def eval(self) -> None:
        self.flownet.eval()

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[int] | None = None,
    ) -> torch.Tensor:
        """Interpolate between img0 and img1 at the given timestep.

        Args:
            img0: (B, 3, H, W) first frame tensor.
            img1: (B, 3, H, W) second frame tensor.
            timestep: Position between frames, 0.0=img0, 1.0=img1.
            scale_list: Multi-scale factors (default [4,2,1]).

        Returns:
            (B, 3, H, W) interpolated frame tensor.
        """
        imgs = torch.cat((img0, img1), dim=1)
        if scale_list is None:
            scale_list = [4, 2, 1]
        return self.flownet(imgs, timestep=timestep, scale_list=scale_list)
