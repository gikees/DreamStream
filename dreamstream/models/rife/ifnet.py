"""IFNet_HDv3 — Practical-RIFE v4.x intermediate flow network (inference only).

Vendored from https://github.com/hzwer/Practical-RIFE (ECCV2022).
Adapted: no Contextnet/Unet refine, device-agnostic, no xformers/flash-attn.
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
    """Single scale block: estimates residual flow + mask at one resolution."""

    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor, scale: int) -> torch.Tensor:
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
                * (1.0 / scale)
            )
            x = torch.cat((x, flow), dim=1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet_HDv3(nn.Module):
    """Multi-scale IFNet: 4 IFBlocks at decreasing channel widths."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[int] | None = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        img0 = x[:, :3]
        img1 = x[:, 3:6]

        # Timestep map: same spatial dims as input, filled with timestep value
        timestep_tensor = (x[:, :1].clone() * 0 + 1) * timestep

        flow = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3]

        for block, scale in zip(blocks, scale_list):
            if flow is None:
                # First block: raw images + timestep, no prior flow
                flow, mask = block(
                    torch.cat((img0, img1, timestep_tensor), dim=1),
                    None,
                    scale=scale,
                )
            else:
                # Subsequent blocks: warped images + timestep + mask, residual on flow
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_d, mask_d = block(
                    torch.cat((warped_img0, warped_img1, timestep_tensor, mask), dim=1),
                    flow,
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
        self.flownet = IFNet_HDv3()
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
            scale_list: Multi-scale factors (default [8,4,2,1]).

        Returns:
            (B, 3, H, W) interpolated frame tensor.
        """
        imgs = torch.cat((img0, img1), dim=1)
        if scale_list is None:
            scale_list = [8, 4, 2, 1]
        return self.flownet(imgs, timestep=timestep, scale_list=scale_list)
