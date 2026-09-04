"""Configuration settings for hand anthropometry capture and landmark measurement."""

from pathlib import Path
from typing import Tuple

# Camera Capture Settings
CAMERA_INDEX: int = 0
CAMERA_FRAME_WIDTH: int = 1280
CAMERA_FRAME_HEIGHT: int = 720
SNAPSHOT_INTERVAL_SECONDS: float = 6.0
FLASH_BANNER_DURATION_SECONDS: float = 1.0

# Storage Directories
BASE_OUTPUT_DIR: Path = Path("snapshots")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection & Visual Settings
HAND_DETECTION_CONFIDENCE: float = 0.6
HAND_TRACKING_CONFIDENCE: float = 0.5
MAX_NUM_HANDS: int = 2

# Visual Colors (BGR)
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)
COLOR_BLUE_ACCENT: Tuple[int, int, int] = (255, 140, 0)
COLOR_ORANGE: Tuple[int, int, int] = (0, 140, 255)
COLOR_GREEN: Tuple[int, int, int] = (50, 220, 50)
COLOR_RED: Tuple[int, int, int] = (40, 40, 230)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)

# Overlay Parameters
POINT_RADIUS: int = 3
POINT_THICKNESS: int = -1
FONT_SCALE_POINT: float = 0.32
FONT_SCALE_HUD: float = 0.65