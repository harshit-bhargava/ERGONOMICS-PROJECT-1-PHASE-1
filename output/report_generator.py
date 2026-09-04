"""Saves clean photo frames and JSON landmark coordinates."""

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
    ) -> Tuple[Path, Path]:
        """Save frame image and keypoint coordinate dictionary."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        img_path = self._output_dir / f"frame_{timestamp_str}_{snapshot_index:04d}.jpg"
        json_path = self._output_dir / f"report_{timestamp_str}_{snapshot_index:04d}.json"

        # Save Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(img_path, format="JPEG", quality=95)

        # Save JSON
        kp_dict = {}
        if keypoints:
            for idx, kp in keypoints.items():
                kp_dict[kp.name] = {
                    "id": kp.id,
                    "x": round(kp.x, 4),
                    "y": round(kp.y, 4),
                    "confidence": round(kp.visibility, 4),
                }

        payload = {
            "timestamp": iso_timestamp,
            "snapshot_index": snapshot_index,
            "body_detected": keypoints is not None,
            "keypoints": kp_dict,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFO] Frame captured: {img_path}")
        return img_path, json_path