"""Geometric calculations for hand segment lengths (L), widths (H), and polygon areas (S)."""

from typing import Dict, List, Tuple
import numpy as np

from pose.keypoint_schema import (
    KeypointData,
    LONGITUDINAL_SEGMENTS,
    SURFACE_POLYGONS,
    TRANSVERSE_WIDTHS,
)


def compute_euclidean_distance(pt_a: KeypointData, pt_b: KeypointData) -> float:
    """Calculate 3D Euclidean distance between two keypoints."""
    return float(np.sqrt((pt_a.x - pt_b.x)**2 + (pt_a.y - pt_b.y)**2 + (pt_a.z - pt_b.z)**2))


def compute_polygon_area_2d(points: List[Tuple[float, float]]) -> float:
    """Compute 2D polygon area using the Shoelace formula."""
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def calculate_hand_metrics(keypoints: Dict[int, KeypointData]) -> Dict[str, Dict[str, float]]:
    """Compute all L lengths, H widths, and S surface metrics from the 50 keypoints."""
    metrics: Dict[str, Dict[str, float]] = {
        "longitudinal_lengths": {},
        "transverse_widths": {},
        "surface_areas": {},
    }

    # 1. Longitudinal lengths L1 - L13
    for name, (id_a, id_b) in LONGITUDINAL_SEGMENTS.items():
        if id_a in keypoints and id_b in keypoints:
            metrics["longitudinal_lengths"][name] = round(
                compute_euclidean_distance(keypoints[id_a], keypoints[id_b]), 4
            )

    # 2. Transverse widths H1 - H10
    for name, (id_a, id_b) in TRANSVERSE_WIDTHS.items():
        if id_a in keypoints and id_b in keypoints:
            metrics["transverse_widths"][name] = round(
                compute_euclidean_distance(keypoints[id_a], keypoints[id_b]), 4
            )

    # 3. Surface polygon areas S1 - S8
    for name, id_list in SURFACE_POLYGONS.items():
        poly_pts = [(keypoints[i].x, keypoints[i].y) for i in id_list if i in keypoints]
        if len(poly_pts) == len(id_list):
            metrics["surface_areas"][name] = round(compute_polygon_area_2d(poly_pts), 6)

    return metrics