"""Vendored RIFE (Real-Time Intermediate Flow Estimation) model.

Architecture: IFNet_HDv3 from Practical-RIFE v4.x (ECCV2022).
Pure PyTorch — no custom CUDA kernels.
"""

from dreamstream.models.rife.ifnet import IFNet_HDv3, Model

__all__ = ["IFNet_HDv3", "Model"]
