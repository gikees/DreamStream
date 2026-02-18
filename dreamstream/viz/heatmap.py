"""Edge-aware uncertainty heatmap generation."""

from __future__ import annotations

import cv2
import numpy as np


def generate_heatmap(
    reconstructed: np.ndarray,
    original: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an uncertainty heatmap for the reconstructed frame.

    With ground truth (original): pixel-diff normalized to COLORMAP_JET.
    Without ground truth: edge-aware uncertainty via distance transform.
    Pixels far from detected edges = higher uncertainty (more likely hallucinated).

    Args:
        reconstructed: BGR frame (upscaled/enhanced output).
        original: Optional ground-truth BGR frame for direct comparison.
        edges: Optional single-channel Canny edge map.

    Returns:
        BGR heatmap of same size as reconstructed.
    """
    h, w = reconstructed.shape[:2]

    if original is not None:
        # Direct pixel difference
        orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC)
        diff = cv2.absdiff(reconstructed, orig_resized)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap

    if edges is not None:
        # Edge-aware uncertainty: distance from edges = uncertainty
        edges_resized = cv2.resize(edges, (w, h), interpolation=cv2.INTER_NEAREST)
        # Invert: edges=255 → 0, background=0 → 255
        inverted = cv2.bitwise_not(edges_resized)
        # Distance transform: how far each pixel is from an edge
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
        # Smooth for visual quality
        smoothed = cv2.GaussianBlur(normalized, (31, 31), 0)
        heatmap = cv2.applyColorMap(smoothed.astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap

    # No edges available — generate edges from the reconstructed frame itself
    gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    auto_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    inverted = cv2.bitwise_not(auto_edges)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    smoothed = cv2.GaussianBlur(normalized, (31, 31), 0)
    heatmap = cv2.applyColorMap(smoothed.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap
