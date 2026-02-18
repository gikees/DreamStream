"""Vendored RIFE (Real-Time Intermediate Flow Estimation) model.

Architecture: IFNet v4.6 from Practical-RIFE.
Pure PyTorch — no custom CUDA kernels.
"""

from dreamstream.models.rife.ifnet import IFNet, Model

__all__ = ["IFNet", "Model"]
