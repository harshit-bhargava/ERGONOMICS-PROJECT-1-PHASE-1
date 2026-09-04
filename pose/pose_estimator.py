"""Robust deep-learning skeletal pose estimator using YOLOv8-Pose."""

from typing import Dict, Optional
import numpy as np
from ultralytics import YOLO

from pose.keypoint_schema import KeypointData
import config


class YOLOSkeletonEstimator:
    """Extracts human skeletal keypoints using YOLOv8-Pose neural network."""

    def __init__(self, model_path: str = "yolov8n-pose.pt") -> None:
        """Load the YOLOv8-Pose weights."""
        self.model = YOLO(model_path)
        self.keypoint_names = {
            0: "Nose",
            1: "L_Eye",
            2: "R_Eye",
            3: "L_Ear",
            4: "R_Ear",
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
        }

    def estimate(self, frame_bgr: np.ndarray) -> Optional[Dict[int, KeypointData]]:
        """Run neural network inference to detect human body keypoints.

        Args:
            frame_bgr: OpenCV input image.

        Returns:
            Dictionary of landmark ID to KeypointData, or None if no person detected.
        """
        results = self.model(frame_bgr, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return None

        kpts_obj = results[0].keypoints
        coords = kpts_obj.xyn[0].cpu().numpy()
        confs = (
            kpts_obj.conf[0].cpu().numpy()
            if kpts_obj.conf is not None
            else np.ones(len(coords))
        )

        if np.mean(confs) < 0.25:
            return None

        keypoints: Dict[int, KeypointData] = {}
        for idx, (x, y) in enumerate(coords):
            keypoints[idx] = KeypointData(
                id=idx,
                name=self.keypoint_names.get(idx, f"P_{idx}"),
                x=float(x),
                y=float(y),
                z=0.0,
                visibility=float(confs[idx]),
            )
        return keypoints

    def release(self) -> None:
        """Release resources."""
        pass