"""Visualizer for complete upper & lower body skeleton, joint angles, and HUD."""

import math
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

import config

# Full body links: Torso, Arms, and Lower Body Legs
BODY_CONNECTIONS = [
    (0, 17),                                  # Head / Neck link
    (17, 5), (17, 6),                         # Neck to Shoulders
    (5, 6),                                   # Shoulder span
    (5, 7), (7, 9),                           # Left Arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                          # Right Arm (Shoulder -> Elbow -> Wrist)
    (5, 11), (6, 12), (11, 12),               # Torso / Hips
    (11, 13), (13, 15),                       # Left Leg (Hip -> Knee -> Ankle)
    (12, 14), (14, 16),                       # Right Leg (Hip -> Knee -> Ankle)
]


class SkeletonVisualizer:
    """Renders complete body connections, joint nodes, angle arcs, and HUD."""

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Dict[int, Any],
        hand_keypoints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        h, w, _ = frame.shape

        def to_px(kp) -> Optional[Tuple[int, int]]:
            if hasattr(kp, "visibility") and kp.visibility > 0.25:
                return (int(kp.x * w), int(kp.y * h))
            return None

        # 1. Draw all skeletal connection lines (including lower body)
        for id_a, id_b in BODY_CONNECTIONS:
            if id_a in keypoints and id_b in keypoints:
                pt_a = to_px(keypoints[id_a])
                pt_b = to_px(keypoints[id_b])
                if pt_a and pt_b:
                    cv2.line(frame, pt_a, pt_b, config.COLOR_CYAN, 2, cv2.LINE_AA)

        # 2. Draw all body keypoint nodes
        for idx, kp in keypoints.items():
            if idx in (1, 2, 3, 4):  # Filter facial eyes/ears
                continue

            pt = to_px(kp)
            if pt:
                joint_color = config.COLOR_GREEN if idx == 17 else config.COLOR_ORANGE
                cv2.circle(frame, pt, 6, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, config.COLOR_BLACK, 1, cv2.LINE_AA)

        return frame

    def draw_joint_arc(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        pt_a: Tuple[int, int],
        pt_b: Tuple[int, int],
        radius: int = 22,
    ) -> None:
        """Draw curved arc for angle rotation."""
        angle1 = math.degrees(math.atan2(pt_a[1] - center[1], pt_a[0] - center[0]))
        angle2 = math.degrees(math.atan2(pt_b[1] - center[1], pt_b[0] - center[0]))

        sweep = (angle2 - angle1) % 360
        if sweep > 180:
            angle1, angle2 = angle2, angle1
            sweep = 360 - sweep

        cv2.ellipse(frame, center, (radius, radius), 0, angle1, angle1 + sweep, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_joint_angles(
        self,
        frame: np.ndarray,
        keypoints: Dict[int, Any],
        joint_angles: Dict[str, float],
        hand_keypoints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        h, w, _ = frame.shape

        def to_px(idx: int) -> Optional[Tuple[int, int]]:
            if idx in keypoints and getattr(keypoints[idx], "visibility", 1.0) > 0.25:
                return (int(keypoints[idx].x * w), int(keypoints[idx].y * h))
            return None

        # Badge position and arc points: (Vertex ID, Point A ID, Point B ID).
        joint_configs = {
            "l_elbow_flexion_deg": (7, 5, 9),
            "r_elbow_flexion_deg": (8, 6, 10),
            "l_upper_arm_relative_to_trunk_deg": (5, 11, 7),
            "r_upper_arm_relative_to_trunk_deg": (6, 12, 8),
            "l_wrist_deviation_deg": (9, 7, None),
            "r_wrist_deviation_deg": (10, 8, None),
            "l_knee_interior_deg": (13, 11, 15),
            "r_knee_interior_deg": (14, 12, 16),
            "l_hip_interior_deg": (11, 5, 13),
            "r_hip_interior_deg": (12, 6, 14),
        }

        for angle_key, angle_val in joint_angles.items():
            if angle_key not in joint_configs:
                continue

            v_id, a_id, b_id = joint_configs[angle_key]
            v_pt = to_px(v_id)
            if not v_pt:
                continue

            a_pt = to_px(a_id) if a_id is not None else None
            b_pt = to_px(b_id) if b_id is not None else None

            # Fallback horizontal vector for wrist labels only.
            if "wrist" in angle_key and a_pt and not b_pt:
                b_pt = (v_pt[0] + 30, v_pt[1])

            # Flexion, signed upper-arm, and wrist-deviation values do not
            # equal the unsigned interior angle shown by a geometric arc.
            display_only_angles = {
                "l_elbow_flexion_deg",
                "r_elbow_flexion_deg",
                "l_upper_arm_relative_to_trunk_deg",
                "r_upper_arm_relative_to_trunk_deg",
                "l_wrist_deviation_deg",
                "r_wrist_deviation_deg",
            }
            if a_pt and b_pt and angle_key not in display_only_angles:
                self.draw_joint_arc(frame, v_pt, a_pt, b_pt, radius=20)

            # Degree badge near joint
            tag_text = f"{angle_val:.0f}°"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

            tx, ty = v_pt[0] + 8, v_pt[1] - 8
            cv2.rectangle(frame, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), config.COLOR_BLACK, -1)
            cv2.rectangle(frame, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), config.COLOR_CYAN, 1)
            cv2.putText(frame, tag_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_WHITE, 1, cv2.LINE_AA)

        # Upper-left Live Telemetry Panel
        if joint_angles:
            card_x, card_y = 20, 75
            card_w, card_h = 210, 25 + (len(joint_angles) * 18)

            overlay = frame.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

            cv2.putText(frame, "LIVE JOINT ANGLES", (card_x + 8, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_CYAN, 1, cv2.LINE_AA)

            for idx, (name, val) in enumerate(joint_angles.items()):
                text = f"{name.replace('_', ' ').title()}: {val:.1f}°"
                row_y = card_y + 34 + (idx * 18)
                cv2.putText(frame, text, (card_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_WHITE, 1, cv2.LINE_AA)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        countdown_sec: float,
        is_flash_active: bool,
        is_tracked: bool,
    ) -> np.ndarray:
        h, w, _ = frame.shape
        status_text = "BODY TRACKED" if is_tracked else "SEARCHING..."
        status_color = config.COLOR_GREEN if is_tracked else (0, 0, 255)

        cv2.rectangle(frame, (20, 20), (280, 60), config.COLOR_BLACK, -1)
        cv2.rectangle(frame, (20, 20), (280, 60), status_color, 1)
        cv2.putText(frame, status_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        timer_text = f"Snapshot: {countdown_sec:.1f}s"
        (tw, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        timer_x = w - tw - 30
        cv2.rectangle(frame, (timer_x - 10, 20), (w - 20, 60), config.COLOR_BLACK, -1)
        cv2.rectangle(frame, (timer_x - 10, 20), (w - 20, 60), config.COLOR_WHITE, 1)
        cv2.putText(frame, timer_text, (timer_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1, cv2.LINE_AA)

        if is_flash_active:
            cv2.rectangle(frame, (0, 0), (w, h), config.COLOR_GREEN, 8)

        cv2.putText(frame, "[Q] Quit  |  [S] Snapshot", (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1, cv2.LINE_AA)
        return frame
