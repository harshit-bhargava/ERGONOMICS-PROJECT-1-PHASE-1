"""Visualizer for skeletal joints, on-body angle arcs, and HUD overlays."""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from pose.keypoint_schema import KeypointData
import config

# Skeletal link pairs (facial landmarks removed; nose connects to synthesized neck)
BODY_CONNECTIONS = [
    (0, 17),                                  # Nose to Neck
    (17, 5), (17, 6),                         # Neck to Shoulders
    (5, 6),                                   # Shoulder width
    (5, 7), (7, 9),                           # Left arm (Shoulder -> Elbow -> Wrist)
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12), (11, 12),               # Torso to Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


class SkeletonVisualizer:
    """Renders body keypoints, on-joint angle arcs, and HUD overlays."""

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict[int, KeypointData]) -> np.ndarray:
        """Draw skeletal link lines and numbered landmark nodes."""
        h, w, _ = frame.shape

        def to_px(kp: KeypointData) -> Tuple[int, int]:
            return int(kp.x * w), int(kp.y * h)

        # 1. Structural skeletal lines
        for id_a, id_b in BODY_CONNECTIONS:
            if id_a in keypoints and id_b in keypoints:
                kp_a, kp_b = keypoints[id_a], keypoints[id_b]
                if kp_a.visibility > 0.35 and kp_b.visibility > 0.35:
                    cv2.line(frame, to_px(kp_a), to_px(kp_b), config.COLOR_CYAN, 2, cv2.LINE_AA)

        # 2. Keypoint nodes
        for idx, kp in keypoints.items():
            if idx in (1, 2, 3, 4):  # Exclude eyes and ears
                continue

            if kp.visibility > 0.35:
                cx, cy = to_px(kp)
                joint_color = config.COLOR_GREEN if idx == 17 else config.COLOR_ORANGE

                cv2.circle(frame, (cx, cy), 6, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, config.COLOR_BLACK, 1, cv2.LINE_AA)

        return frame

    def draw_angle_arc(
        self,
        frame: np.ndarray,
        pt_a: Tuple[int, int],
        pt_b: Tuple[int, int],
        pt_c: Tuple[int, int],
        angle_val: float,
        radius: int = 45,
    ) -> None:
        """Draw a circular angle arc at vertex B with degree text."""
        # Vectors: Vertex B -> A, and Vertex B -> C
        v_ba = np.array([pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]], dtype=np.float64)
        v_bc = np.array([pt_c[0] - pt_b[0], pt_c[1] - pt_b[1]], dtype=np.float64)

        if np.linalg.norm(v_ba) < 1e-5 or np.linalg.norm(v_bc) < 1e-5:
            return

        ang_a = np.arctan2(v_ba[1], v_ba[0])
        ang_c = np.arctan2(v_bc[1], v_bc[0])

        # Calculate shortest angular sweep between vectors
        diff = (ang_c - ang_a) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi

        # Generate arc points
        angles = np.linspace(ang_a, ang_a + diff, 30)
        arc_points = np.array(
            [
                [int(pt_b[0] + radius * np.cos(a)), int(pt_b[1] + radius * np.sin(a))]
                for a in angles
            ],
            dtype=np.int32,
        )

        # Draw black outline and colored curve for high visibility
        cv2.polylines(frame, [arc_points], isClosed=False, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [arc_points], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Place degree text label along the midpoint angle
        mid_angle = ang_a + diff / 2.0
        text_radius = radius + 28
        tx = int(pt_b[0] + text_radius * np.cos(mid_angle))
        ty = int(pt_b[1] + text_radius * np.sin(mid_angle))

        label = f"{int(round(angle_val))}°"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)

        # Text shadow/background
        cv2.putText(
            frame,
            label,
            (tx - lw // 2, ty + lh // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        # Foreground degree text
        cv2.putText(
            frame,
            label,
            (tx - lw // 2, ty + lh // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_joint_angles(
        self,
        frame: np.ndarray,
        keypoints: Dict[int, KeypointData],
        joint_angles: Dict[str, float],
    ) -> np.ndarray:
        """Render on-body arcs and the telemetry card."""
        h, w, _ = frame.shape

        def to_px(idx: int) -> Optional[Tuple[int, int]]:
            if idx in keypoints and keypoints[idx].visibility > 0.35:
                return (int(keypoints[idx].x * w), int(keypoints[idx].y * h))
            return None

        # Draw Left Elbow Arc (5: L_Shoulder, 7: L_Elbow, 9: L_Wrist)
        if "l_elbow" in joint_angles:
            p_sho, p_elb, p_wri = to_px(5), to_px(7), to_px(9)
            if p_sho and p_elb and p_wri:
                self.draw_angle_arc(frame, p_sho, p_elb, p_wri, joint_angles["l_elbow"])

        # Draw Right Elbow Arc (6: R_Shoulder, 8: R_Elbow, 10: R_Wrist)
        if "r_elbow" in joint_angles:
            p_sho, p_elb, p_wri = to_px(6), to_px(8), to_px(10)
            if p_sho and p_elb and p_wri:
                self.draw_angle_arc(frame, p_sho, p_elb, p_wri, joint_angles["r_elbow"])

        # Telemetry Card (top-left)
        if joint_angles:
            card_x, card_y = 20, 75
            card_w, card_h = 220, 25 + (len(joint_angles) * 20)

            overlay = frame.copy()
            cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

            cv2.putText(frame, "LIVE JOINT ANGLES", (card_x + 10, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_CYAN, 1, cv2.LINE_AA)

            for i, (name, val) in enumerate(joint_angles.items()):
                clean_name = name.replace("_", " ").title()
                cv2.putText(
                    frame,
                    f"{clean_name}: {val:.1f}°",
                    (card_x + 12, card_y + 36 + (i * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    config.COLOR_WHITE,
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        countdown_sec: float,
        is_flash_active: bool,
        is_tracked: bool,
    ) -> np.ndarray:
        """Render status banner, capture flash, and countdown timer."""
        h, w, _ = frame.shape

        status_text = "BODY TRACKED" if is_tracked else "SEARCHING FOR PERSON..."
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