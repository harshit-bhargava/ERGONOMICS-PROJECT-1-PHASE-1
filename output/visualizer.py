"""Visualizer for skeletal joints, clean connections, and capture HUD."""

from typing import Dict
import cv2
import numpy as np

from pose.keypoint_schema import KeypointData
import config

# Skeletal link pairs: no eyes or ears; head connects directly Nose -> Neck -> Shoulders
BODY_CONNECTIONS = [
    (0, 17),                                  # Head / Cervical spine: Nose to Neck
    (17, 5), (17, 6),                         # Neck to Left & Right Shoulders
    (5, 6),                                   # Shoulder span
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12), (11, 12),               # Torso / Spine to Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


class SkeletonVisualizer:
    """Renders clean body keypoints, links, and capture interface."""

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict[int, KeypointData]) -> np.ndarray:
        """Draw skeleton links and numbered keypoint nodes."""
        h, w, _ = frame.shape

        def to_px(kp: KeypointData):
            return int(kp.x * w), int(kp.y * h)

        # 1. Draw structural skeletal lines
        for id_a, id_b in BODY_CONNECTIONS:
            if id_a in keypoints and id_b in keypoints:
                kp_a, kp_b = keypoints[id_a], keypoints[id_b]
                if kp_a.visibility > 0.35 and kp_b.visibility > 0.35:
                    cv2.line(frame, to_px(kp_a), to_px(kp_b), config.COLOR_CYAN, 2, cv2.LINE_AA)

        # 2. Draw numbered landmark circles
        for idx, kp in keypoints.items():
            # Enforce exclusion of facial peripherals
            if idx in (1, 2, 3, 4):
                continue

            if kp.visibility > 0.35:
                cx, cy = to_px(kp)
                joint_color = config.COLOR_GREEN if idx == 17 else config.COLOR_ORANGE

                cv2.circle(frame, (cx, cy), 6, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, config.COLOR_BLACK, 1, cv2.LINE_AA)

                label = f"{idx}:{kp.name}"
                cv2.putText(
                    frame,
                    label,
                    (cx + 8, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    config.COLOR_WHITE,
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def draw_hud(self, frame: np.ndarray, countdown_sec: float, is_flash_active: bool, is_tracked: bool) -> np.ndarray:
        """Render status and capture flash banner."""
        h, w, _ = frame.shape

        # Detection Status Badge
        status_text = "BODY & NECK DETECTED" if is_tracked else "SEARCHING FOR PERSON..."
        status_color = config.COLOR_GREEN if is_tracked else (0, 0, 255)
        cv2.rectangle(frame, (20, 20), (360, 60), config.COLOR_BLACK, -1)
        cv2.rectangle(frame, (20, 20), (360, 60), status_color, 1)
        cv2.putText(frame, status_text, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # Countdown Timer
        timer_text = f"Next capture: {countdown_sec:.1f}s"
        (tw, _), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        timer_x = w - tw - 30
        cv2.rectangle(frame, (timer_x - 10, 20), (w - 20, 60), config.COLOR_BLACK, -1)
        cv2.rectangle(frame, (timer_x - 10, 20), (w - 20, 60), config.COLOR_WHITE, 1)
        cv2.putText(frame, timer_text, (timer_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_WHITE, 1, cv2.LINE_AA)

        # Snapshot Flash Effect
        if is_flash_active:
            cv2.rectangle(frame, (0, 0), (w, h), config.COLOR_GREEN, 10)
            banner = "SNAPSHOT CAPTURED"
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx1, by1 = (w - bw) // 2 - 20, (h - bh) // 2 - 20
            cv2.rectangle(frame, (bx1, by1), (bx1 + bw + 40, by1 + bh + 40), config.COLOR_BLACK, -1)
            cv2.rectangle(frame, (bx1, by1), (bx1 + bw + 40, by1 + bh + 40), config.COLOR_GREEN, 2)
            cv2.putText(frame, banner, (bx1 + 20, by1 + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_GREEN, 2, cv2.LINE_AA)

        cv2.putText(frame, "[Q] Quit  |  [S] Save Snapshot", (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1, cv2.LINE_AA)
        return frame