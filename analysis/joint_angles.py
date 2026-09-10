"""Vector mathematics for calculating ergonomic joint angles from body keypoints."""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def calculate_3point_angle(
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    pt_c: Tuple[float, float],
) -> float:
    """Calculate the planar angle ∠ABC at vertex B in degrees."""
    vec_ba = np.array([pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]], dtype=np.float64)
    vec_bc = np.array([pt_c[0] - pt_b[0], pt_c[1] - pt_b[1]], dtype=np.float64)

    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def compute_all_joint_angles(keypoints: Dict[int, Any]) -> Dict[str, float]:
    """Calculate critical ergonomic joint angles with desk/webcam fallbacks."""
    angles: Dict[str, float] = {}

    def get_pt(idx: int) -> Optional[Tuple[float, float]]:
        if idx in keypoints and keypoints[idx].visibility > 0.30:
            return (keypoints[idx].x, keypoints[idx].y)
        return None

    nose = get_pt(0)
    neck = get_pt(17)
    l_sho, r_sho = get_pt(5), get_pt(6)
    l_elb, r_elb = get_pt(7), get_pt(8)
    l_wri, r_wri = get_pt(9), get_pt(10)
    l_hip, r_hip = get_pt(11), get_pt(12)
    l_kne, r_kne = get_pt(13), get_pt(14)
    l_ank, r_ank = get_pt(15), get_pt(16)

    # 1. Elbow Flexion (Shoulder -> Elbow -> Wrist)
    if l_sho and l_elb and l_wri:
        angles["l_elbow"] = round(calculate_3point_angle(l_sho, l_elb, l_wri), 1)
    if r_sho and r_elb and r_wri:
        angles["r_elbow"] = round(calculate_3point_angle(r_sho, r_elb, r_wri), 1)

    # 2. Shoulder / Upper Arm Elevation (Hip -> Shoulder -> Elbow)
    # Fallback to vertical downward axis if hips are below camera frame
    if l_sho and l_elb:
        ref_down = l_hip if l_hip else (l_sho[0], l_sho[1] + 0.5)
        angles["l_shoulder"] = round(calculate_3point_angle(ref_down, l_sho, l_elb), 1)

    if r_sho and r_elb:
        ref_down = r_hip if r_hip else (r_sho[0], r_sho[1] + 0.5)
        angles["r_shoulder"] = round(calculate_3point_angle(ref_down, r_sho, r_elb), 1)

    # 3. Neck Flexion (Nose -> Neck -> Trunk Base)
    # Works even if hips are completely hidden by desk
    if nose and neck:
        trunk_base = (
            l_hip if l_hip else (r_hip if r_hip else (neck[0], neck[1] + 0.5))
        )
        raw_angle = calculate_3point_angle(nose, neck, trunk_base)
        # Cervical deviation: 0° is straight vertical, >20° is forward bend
        angles["neck"] = round(abs(180.0 - raw_angle), 1)

    # 4. Knee Bend (Hip -> Knee -> Ankle)
    if l_hip and l_kne and l_ank:
        angles["l_knee"] = round(calculate_3point_angle(l_hip, l_kne, l_ank), 1)
    if r_hip and r_kne and r_ank:
        angles["r_knee"] = round(calculate_3point_angle(r_hip, r_kne, r_ank), 1)

    return angles