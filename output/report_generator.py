"""Saves photo frames and JSON reports containing keypoint coordinates and joint angles."""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from pose.keypoint_schema import KeypointData
import config


class ReportGenerator:
    """Manages snapshot and JSON persistence."""

    def __init__(self, output_dir: Path = config.BASE_OUTPUT_DIR) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        frame_bgr: np.ndarray,
        snapshot_index: int,
        keypoints: Optional[Dict[int, KeypointData]],
        joint_angles: Dict[str, float],
    ) -> Tuple[Path, Path]:
        """Save frame image and keypoint + angle report."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        img_path = self._output_dir / f"frame_{timestamp_str}_{snapshot_index:04d}.jpg"
        json_path = self._output_dir / f"report_{timestamp_str}_{snapshot_index:04d}.json"

        # 1. Save Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(img_path, format="JPEG", quality=95)

        # 2. Format Keypoints
        kp_dict = {}
        if keypoints:
            for idx, kp in keypoints.items():
                if idx in (1, 2, 3, 4):  # Exclude facial peripherals
                    continue
                kp_dict[kp.name] = {
                    "id": kp.id,
                    "x": round(kp.x, 4),
                    "y": round(kp.y, 4),
                    "confidence": round(kp.visibility, 4),
                }

        # 3. Assemble JSON Payload
        payload = {
            "timestamp": iso_timestamp,
            "snapshot_index": snapshot_index,
            "body_detected": keypoints is not None,
            "joint_angles": joint_angles,
            "keypoints": kp_dict,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFO] Snapshot #{snapshot_index} saved with {len(joint_angles)} computed angles.")
        return img_path, json_path

def save_snapshot(
        self,
        frame_bgr: np.ndarray,
        snapshot_index: int,
        keypoints: Optional[Dict[int, KeypointData]],
        joint_angles: Optional[Dict[str, float]] = None,
    ) -> Tuple[Path, Path]:
        """Save frame image and keypoint coordinate dictionary with joint angles."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        img_path = self._output_dir / f"frame_{timestamp_str}_{snapshot_index:04d}.jpg"
        json_path = self._output_dir / f"report_{timestamp_str}_{snapshot_index:04d}.json"

        # Save Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(img_path, format="JPEG", quality=95)

        # Process Keypoints
        kp_dict = {}
        if keypoints:
            for idx, kp in keypoints.items():
                kp_dict[kp.name] = {
                    "id": kp.id,
                    "x": round(kp.x, 4),
                    "y": round(kp.y, 4),
                    "confidence": round(kp.visibility, 4),
                }

        # Build payload with joint angles
        payload = {
            "timestamp": iso_timestamp,
            "snapshot_index": snapshot_index,
            "body_detected": keypoints is not None,
            "keypoints": kp_dict,
            "joint_angles": joint_angles or {},
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFO] Frame captured: {img_path}")
        return img_path, json_path