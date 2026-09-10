"""Configuration settings for YOLOv8 skeletal tracking and ergonomic risk assessment."""

from pathlib import Path
from typing import Tuple, Union

# Camera & Video Stream Settings
CAMERA_INDEX: Union[int, str] = 0  # Use 0 for webcam, or "video.mp4" for recorded task footage
CAMERA_FRAME_WIDTH: int = 1280
CAMERA_FRAME_HEIGHT: int = 720
WINDOW_TITLE: str = "Real-Time Ergonomic Pose & Joint Angle System"

# Snapshot & Capture Settings
SNAPSHOT_INTERVAL_SECONDS: float = 6.0
SNAPSHOT_INTERVAL_SEC: float = SNAPSHOT_INTERVAL_SECONDS  # Backward compatibility alias
FLASH_BANNER_DURATION_SECONDS: float = 1.0

# Storage Directories
BASE_OUTPUT_DIR: Path = Path("snapshots")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pose Model & Confidence Thresholds
POSE_MODEL_PATH: str = "yolov8n-pose.pt"
POSE_DETECTION_CONFIDENCE: float = 0.25
KEYPOINT_VISIBILITY_THRESHOLD: float = 0.35

# Visual Colors (BGR Format for OpenCV)
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)
COLOR_BLUE_ACCENT: Tuple[int, int, int] = (255, 140, 0)
COLOR_ORANGE: Tuple[int, int, int] = (0, 140, 255)
COLOR_GREEN: Tuple[int, int, int] = (50, 220, 50)
COLOR_RED: Tuple[int, int, int] = (40, 40, 230)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)

# Overlay & Font Parameters
POINT_RADIUS: int = 6
LINE_THICKNESS: int = 2
FONT_SCALE_LABEL: float = 0.38
FONT_SCALE_ANGLE: float = 0.45
FONT_SCALE_HUD: float = 0.60