"""Vector mathematics for calculating ergonomic joint angles from YOLO skeletal keypoints."""

from typing import Dict, Optional, Tuple
import numpy as np

from pose.keypoint_schema import KeypointData


def calculate_3point_angle(
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    pt_c: Tuple[float, float],
) -> float:
    """Calculate the planar angle ∠ABC at vertex B in degrees.

    Args:
        pt_a: First point (x, y).
        pt_b: Vertex point (x, y).
        pt_c: Third point (x, y).

    Returns:
        Included angle in degrees [0.0, 180.0].
    """
    vec_ba = np.array([pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]], dtype=np.float64)
    vec_bc = np.array([pt_c[0] - pt_b[0], pt_c[1] - pt_b[1]], dtype=np.float64)

    norm_ba = np.linalg.norm(vec_ba)
    norm_bc = np.linalg.norm(vec_bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_vector_inclination_from_vertical(
    origin: Tuple[float, float],
    target: Tuple[float, float],
) -> float:
    """Compute the angle of deviation of a segment from the vertical upright (0, -1).

    Args:
        origin: Bottom pivot point (x, y), e.g., hip.
        target: Top point (x, y), e.g., neck/shoulder.

    Returns:
        Angular deviation from vertical in degrees [0.0, 180.0].
    """
    segment_vec = np.array([target[0] - origin[0], target[1] - origin[1]], dtype=np.float64)
    vertical_up_vec = np.array([0.0, -1.0], dtype=np.float64)

    norm_seg = np.linalg.norm(segment_vec)
    if norm_seg < 1e-6:
        return 0.0

    cosine_val = np.dot(segment_vec, vertical_up_vec) / norm_seg
    cosine_val = np.clip(cosine_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_val)))


def compute_all_joint_angles(keypoints: Dict[int, KeypointData]) -> Dict[str, float]:
    """Compute ergonomic joint angles from detected body landmarks.

    Keypoint IDs:
        0: Nose, 17: Neck
        5: L_Shoulder, 6: R_Shoulder
        7: L_Elbow, 8: R_Elbow
        9: L_Wrist, 10: R_Wrist
        11: L_Hip, 12: R_Hip
        13: L_Knee, 14: R_Knee
        15: L_Ankle, 16: R_Ankle
    """
    angles: Dict[str, float] = {}

    def get_pt(idx: int) -> Optional[Tuple[float, float]]:
        kp = keypoints.get(idx)
        if kp and kp.visibility > 0.35:
            return (kp.x, kp.y)
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
        angles["left_elbow_flexion_deg"] = round(calculate_3point_angle(l_sho, l_elb, l_wri), 1)
    if r_sho and r_elb and r_wri:
        angles["right_elbow_flexion_deg"] = round(calculate_3point_angle(r_sho, r_elb, r_wri), 1)

    # 2. Shoulder / Upper Arm Flexion (Hip -> Shoulder -> Elbow)
    if l_hip and l_sho and l_elb:
        angles["left_shoulder_lift_deg"] = round(calculate_3point_angle(l_hip, l_sho, l_elb), 1)
    if r_hip and r_sho and r_elb:
        angles["right_shoulder_lift_deg"] = round(calculate_3point_angle(r_hip, r_sho, r_elb), 1)

    # 3. Knee Flexion (Hip -> Knee -> Ankle)
    if l_hip and l_kne and l_ank:
        angles["left_knee_flexion_deg"] = round(calculate_3point_angle(l_hip, l_kne, l_ank), 1)
    if r_hip and r_kne and r_ank:
        angles["right_knee_flexion_deg"] = round(calculate_3point_angle(r_hip, r_kne, r_ank), 1)

    # Midpoint of hips for trunk calculations
    hip_mid = None
    if l_hip and r_hip:
        hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
    elif l_hip:
        hip_mid = l_hip
    elif r_hip:
        hip_mid = r_hip

    # 4. Trunk Inclination / Flexion (Deviation of spine from vertical)
    if hip_mid and neck:
        angles["trunk_flexion_deg"] = round(
            calculate_vector_inclination_from_vertical(origin=hip_mid, target=neck), 1
        )

    # 5. Neck Flexion (Nose -> Neck relative to Trunk axis)
    if nose and neck and hip_mid:
        cervical_angle = calculate_3point_angle(nose, neck, hip_mid)
        # Deviation from straight alignment (180 degrees)
        angles["neck_flexion_deg"] = round(abs(180.0 - cervical_angle), 1)

    return angles