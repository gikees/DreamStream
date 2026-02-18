"""IFNet_HDv3 — Practical-RIFE v4.26 intermediate flow network (inference only).

Vendored from https://github.com/hzwer/Practical-RIFE (v4.26, 2024.09.21).
Architecture: Head encoder + 5 IFBlocks with ResConv and PixelShuffle output.
Adapted: device-agnostic, no xformers/flash-attn, teacher/contextnet removed.
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
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    """Lightweight feature encoder: 3-channel image -> 4-channel features."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.cnn3(x)
        return x


class ResConv(nn.Module):
    """Residual conv block with learnable scale."""

    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Multi-scale block: flow + mask + feature output via PixelShuffle."""

    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        # Output: 4*13 channels -> PixelShuffle(2) -> 13 channels at 2x spatial
        # 13 = 4 (flow) + 1 (mask) + 8 (features)
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None = None, scale: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
            x = torch.cat((x, flow), dim=1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    """IFNet_HDv3 v4.26: Head encoder + 5 IFBlocks."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[int] | None = None,
    ) -> torch.Tensor:
        if scale_list is None:
            scale_list = [8, 4, 2, 1, 1]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow = None
        mask = None
        feat = None
        warped_img0 = img0
        warped_img1 = img1

        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i, (block, scale) in enumerate(zip(blocks, scale_list)):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), dim=1),
                    None,
                    scale=scale,
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = block(
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask, feat), dim=1),
                    flow,
                    scale=scale,
                )
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)


class Model:
    """Wrapper matching the API expected by RIFEInterpolator."""

    def __init__(self) -> None:
        self.flownet = IFNet()
        self.device = torch.device("cpu")

    def load_model(self, path: str, rank: int = 0) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / "flownet.pkl"

        self.device = torch.device("cpu") if rank < 0 else torch.device("cuda")

        state = torch.load(p, map_location="cpu", weights_only=True)
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
        imgs = torch.cat((img0, img1), dim=1)
        if scale_list is None:
            scale_list = [8, 4, 2, 1, 1]
        return self.flownet(imgs, timestep=timestep, scale_list=scale_list)
