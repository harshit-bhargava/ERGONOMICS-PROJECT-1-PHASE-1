"""Deep-learning skeletal pose estimator with synthesized neck landmark and excluded facial peripherals (eyes and ears)."""

from typing import Dict, Optional
import numpy as np
from ultralytics import YOLO

from pose.keypoint_schema import KeypointData
import config


class YOLOSkeletonEstimator:
    """Extracts human skeletal keypoints, synthesizes neck, and removes eyes and ears."""

    def __init__(self, model_path: str = "yolov8n-pose.pt") -> None:
        """Load YOLOv8-Pose model."""
        self.model = YOLO(model_path)
        # 1, 2 (Eyes) and 3, 4 (Ears) completely removed; 17 added as Neck
        self.keypoint_names = {
            0: "Nose",
            5: "L_Shoulder",
            6: "R_Shoulder",
            7: "L_Elbow",
            8: "R_Elbow",
            9: "L_Wrist",
            10: "R_Wrist",
            11: "L_Hip",
            12: "R_Hip",
            13: "L_Knee",
            14: "R_Knee",
            15: "L_Ankle",
            16: "R_Ankle",
            17: "Neck",
        }

    def estimate(self, frame_bgr: np.ndarray) -> Optional[Dict[int, KeypointData]]:
        """Run pose inference and construct the ergonomic skeletal dictionary."""
        results = self.model(frame_bgr, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return None

        kpts_obj = results[0].keypoints
        coords = kpts_obj.xyn[0].cpu().numpy()  # Normalized coordinates [17, 2]
        confs = (
            kpts_obj.conf[0].cpu().numpy()
            if kpts_obj.conf is not None
            else np.ones(len(coords))
        )

        if np.mean(confs) < 0.25:
            return None

        keypoints: Dict[int, KeypointData] = {}

        # 1. Populate body keypoints, skipping eyes (1, 2) and ears (3, 4)
        for idx in range(len(coords)):
            if idx in (1, 2, 3, 4):
                continue
            name = self.keypoint_names.get(idx, f"P_{idx}")
            keypoints[idx] = KeypointData(
                id=idx,
                name=name,
                x=float(coords[idx][0]),
                y=float(coords[idx][1]),
                z=0.0,
                visibility=float(confs[idx]),
            )

        # 2. Synthesize Neck landmark (ID: 17) as midpoint of Shoulders
        l_sho_conf = float(confs[5])
        r_sho_conf = float(confs[6])

        if l_sho_conf > 0.25 and r_sho_conf > 0.25:
            neck_x = float((coords[5][0] + coords[6][0]) / 2.0)
            neck_y = float((coords[5][1] + coords[6][1]) / 2.0)
            neck_conf = float((l_sho_conf + r_sho_conf) / 2.0)
        elif l_sho_conf > 0.25:
            neck_x, neck_y, neck_conf = float(coords[5][0]), float(coords[5][1]), l_sho_conf
        elif r_sho_conf > 0.25:
            neck_x, neck_y, neck_conf = float(coords[6][0]), float(coords[6][1]), r_sho_conf
        else:
            neck_x, neck_y, neck_conf = 0.0, 0.0, 0.0

        if neck_conf > 0.25:
            keypoints[17] = KeypointData(
                id=17,
                name="Neck",
                x=neck_x,
                y=neck_y,
                z=0.0,
                visibility=neck_conf,
            )

        return keypoints

    def release(self) -> None:
        """Release allocated resources."""
        pass