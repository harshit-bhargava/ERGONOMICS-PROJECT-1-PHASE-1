"""MediaPipe hand landmarks matched to YOLO body wrists."""

from itertools import combinations, permutations
from typing import Any, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


Point = Tuple[float, float]


class MediaPipeHandTracker:
    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        body_keypoints: Dict[int, Any] | None = None,
    ) -> Dict[str, Point]:
        """Return normalized MCP points assigned to visible left/right wrists."""
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection = self._hands.process(frame_rgb)

        if not detection.multi_hand_landmarks:
            return {}

        hands = []
        for landmarks in detection.multi_hand_landmarks:
            wrist = landmarks.landmark[0]
            middle_mcp = landmarks.landmark[9]
            hands.append(
                {
                    "wrist": (float(wrist.x), float(wrist.y)),
                    "mcp": (float(middle_mcp.x), float(middle_mcp.y)),
                }
            )

        if not body_keypoints:
            return {}

        body_wrists = []
        for side, idx in (("l", 9), ("r", 10)):
            kp = body_keypoints.get(idx)
            if kp is not None and getattr(kp, "visibility", 0.0) >= 0.50:
                body_wrists.append((side, (float(kp.x), float(kp.y))))

        if not body_wrists:
            return {}

        # Find the closest valid one-to-one assignment. This checks either
        # body side when only one hand is detected.
        max_match_distance = 0.12
        match_count = min(len(body_wrists), len(hands))
        best_assignment = None
        best_distance = float("inf")

        for wrist_subset in combinations(body_wrists, match_count):
            for hand_indices in permutations(range(len(hands)), match_count):
                pairs = list(zip(wrist_subset, hand_indices))
                distances = [
                    float(
                        np.hypot(
                            body_point[0] - hands[hand_index]["wrist"][0],
                            body_point[1] - hands[hand_index]["wrist"][1],
                        )
                    )
                    for (_, body_point), hand_index in pairs
                ]

                # Reject assignments where a hand is too far from its
                # corresponding YOLO wrist.
                if any(distance > max_match_distance for distance in distances):
                    continue

                total_distance = sum(distances)
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_assignment = pairs

        if best_assignment is None:
            return {}

        output: Dict[str, Point] = {}
        for (side, _body_point), hand_index in best_assignment:
            output[f"{side}_finger_mcp"] = hands[hand_index]["mcp"]

        return output

    def release(self) -> None:
        self._hands.close()
