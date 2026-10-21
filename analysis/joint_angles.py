"""2D posture measurements from normalized pose and hand landmarks.

Angles are image-plane estimates. Use a side view for forward/backward
posture measurements and a front view for left/right deviation.
"""

from typing import Any, Dict, Optional, Tuple
import math


Point = Tuple[float, float]


def calculate_3point_angle(a: Point, b: Point, c: Point) -> float:
    """Return the unsigned interior angle ABC in degrees."""
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    len_ba = math.hypot(bax, bay)
    len_bc = math.hypot(bcx, bcy)

    if len_ba < 1e-8 or len_bc < 1e-8:
        return 0.0

    cosine = (bax * bcx + bay * bcy) / (len_ba * len_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _signed_angle_between(a: Point, b: Point) -> float:
    """Signed angle from vector a to vector b; positive is counterclockwise."""
    ax, ay = a[0], -a[1]  # Convert image y-down coordinates to y-up.
    bx, by = b[0], -b[1]

    if math.hypot(ax, ay) < 1e-8 or math.hypot(bx, by) < 1e-8:
        return 0.0

    cross = ax * by - ay * bx
    dot = ax * bx + ay * by
    return math.degrees(math.atan2(cross, dot))


def _get_point(keypoints: Dict[int, Any], idx: int) -> Optional[Point]:
    kp = keypoints.get(idx)
    if kp is None or getattr(kp, "visibility", 0.0) < 0.50:
        return None
    return float(kp.x), float(kp.y)


def _midpoint(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _add_angle(result: Dict[str, float], name: str, value: float) -> None:
    result[name] = round(value, 1)


def compute_all_joint_angles(
    keypoints: Dict[int, Any],
    hand_keypoints: Optional[Dict[str, Point]] = None,
) -> Dict[str, float]:
    """Calculate available 2D posture measurements.

    Keypoint IDs follow the COCO YOLO pose order:
    nose 0, ears 3/4, shoulders 5/6, elbows 7/8, wrists 9/10,
    hips 11/12, knees 13/14, ankles 15/16.
    """
    result: Dict[str, float] = {}

    nose = _get_point(keypoints, 0)
    left_ear = _get_point(keypoints, 3)
    right_ear = _get_point(keypoints, 4)

    left_shoulder = _get_point(keypoints, 5)
    right_shoulder = _get_point(keypoints, 6)
    left_elbow = _get_point(keypoints, 7)
    right_elbow = _get_point(keypoints, 8)
    left_wrist = _get_point(keypoints, 9)
    right_wrist = _get_point(keypoints, 10)
    left_hip = _get_point(keypoints, 11)
    right_hip = _get_point(keypoints, 12)
    left_knee = _get_point(keypoints, 13)
    right_knee = _get_point(keypoints, 14)
    left_ankle = _get_point(keypoints, 15)
    right_ankle = _get_point(keypoints, 16)

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    ear_mid = _midpoint(left_ear, right_ear)

    # Trunk inclination relative to the image's vertical axis.
    # This works best when the camera is level and the worker is side-on.
    if shoulder_mid is not None and hip_mid is not None:
        trunk_up = (
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1],
        )
        _add_angle(
            result,
            "trunk_inclination_from_vertical_deg",
            math.degrees(math.atan2(trunk_up[0], -trunk_up[1])),
        )

    # A head orientation proxy, not a direct neck-joint angle.
    # The nose-to-ear direction is compared with the trunk direction.
    if nose is not None and ear_mid is not None and shoulder_mid is not None and hip_mid is not None:
        head_forward = (nose[0] - ear_mid[0], nose[1] - ear_mid[1])
        trunk_up = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        _add_angle(
            result,
            "head_to_trunk_relative_deg",
            _signed_angle_between(trunk_up, head_forward),
        )

    # Upper-arm angle relative to the shoulder-to-hip trunk line.
    # Sign depends on camera side and image orientation; retain it for review.
    for side, shoulder, elbow, hip in (
        ("l", left_shoulder, left_elbow, left_hip),
        ("r", right_shoulder, right_elbow, right_hip),
    ):
        if shoulder is not None and elbow is not None and hip is not None:
            trunk_down = (hip[0] - shoulder[0], hip[1] - shoulder[1])
            upper_arm = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
            _add_angle(
                result,
                f"{side}_upper_arm_relative_to_trunk_deg",
                _signed_angle_between(trunk_down, upper_arm),
            )

        # Elbow interior angle and flexion angle are both useful:
        # flexion is 180 degrees minus the interior angle.
        wrist = left_wrist if side == "l" else right_wrist
        if shoulder is not None and elbow is not None and wrist is not None:
            interior = calculate_3point_angle(shoulder, elbow, wrist)
            _add_angle(result, f"{side}_elbow_interior_deg", interior)
            _add_angle(result, f"{side}_elbow_flexion_deg", 180.0 - interior)

        # Wrist deviation from a straight forearm-to-hand line.
        if hand_keypoints and elbow is not None and wrist is not None:
            mcp = hand_keypoints.get(f"{side}_finger_mcp")
            if mcp is not None:
                interior = calculate_3point_angle(elbow, wrist, mcp)
                _add_angle(result, f"{side}_wrist_deviation_deg", 180.0 - interior)

    # Hip and knee interior angles.
    for side, hip, knee, ankle in (
        ("l", left_hip, left_knee, left_ankle),
        ("r", right_hip, right_knee, right_ankle),
    ):
        if hip is not None and knee is not None and ankle is not None:
            _add_angle(
                result,
                f"{side}_knee_interior_deg",
                calculate_3point_angle(hip, knee, ankle),
            )

        shoulder = left_shoulder if side == "l" else right_shoulder
        if shoulder is not None and hip is not None and knee is not None:
            _add_angle(
                result,
                f"{side}_hip_interior_deg",
                calculate_3point_angle(shoulder, hip, knee),
            )

    return result