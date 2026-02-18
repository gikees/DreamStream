"""Sender simulation — degrades video to ultra-low-bandwidth hints."""

from dreamstream.sender_sim.degrader import degrade_video, probe_video
from dreamstream.sender_sim.hints import compute_canny_edges, edges_to_rgb

__all__ = ["degrade_video", "probe_video", "compute_canny_edges", "edges_to_rgb"]
