"""OpenCV webcam video capture and time-based frame snapshot trigger."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

import config


class CameraHandler:
    """Manages video capture device and frame snapshot timing."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        frame_width: int = config.CAMERA_FRAME_WIDTH,
        frame_height: int = config.CAMERA_FRAME_HEIGHT,
        snapshot_interval_sec: float = config.SNAPSHOT_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the video capture device and timing parameters.

        Args:
            camera_index: Hardware video device index.
            frame_width: Desired capture frame width.
            frame_height: Desired capture frame height.
            snapshot_interval_sec: Interval in seconds between auto snapshots.
        """
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self._snapshot_interval = snapshot_interval_sec
        self._last_snapshot_time = time.time()
        self._last_capture_event_time: float = -10.0
        self._snapshot_count: int = 0
        self._force_snapshot_flag: bool = False

    def is_opened(self) -> bool:
        """Check if camera device is opened and ready.

        Returns:
            True if device is open, False otherwise.
        """
        return self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera stream.

        Returns:
            Tuple of (success_boolean, bgr_image_array).
        """
        success, frame = self._cap.read()
        return success, frame

    def should_capture(self) -> bool:
        """Evaluate if an automatic or manual snapshot is triggered.

        Returns:
            True if snapshot interval elapsed or manual flag set, False otherwise.
        """
        current_time = time.time()
        elapsed = current_time - self._last_snapshot_time

        if self._force_snapshot_flag or (elapsed >= self._snapshot_interval):
            self._last_snapshot_time = current_time
            self._last_capture_event_time = current_time
            self._snapshot_count += 1
            self._force_snapshot_flag = False
            return True
        return False

    def trigger_manual_snapshot(self) -> None:
        """Request an immediate manual snapshot capture on next cycle."""
        self._force_snapshot_flag = True

    def get_countdown_seconds(self) -> float:
        """Calculate remaining time until the next automatic snapshot.

        Returns:
            Remaining seconds as float [0.0, snapshot_interval].
        """
        elapsed = time.time() - self._last_snapshot_time
        remaining = max(0.0, self._snapshot_interval - elapsed)
        return remaining

    def is_in_flash_state(self) -> bool:
        """Check if visual snapshot capture indicator should be active.

        Returns:
            True if current time is within flash banner duration after capture.
        """
        return (time.time() - self._last_capture_event_time) < config.FLASH_BANNER_DURATION_SECONDS

    @property
    def snapshot_index(self) -> int:
        """Get the current snapshot sequence count.

        Returns:
            Total snapshots taken so far.
        """
        return self._snapshot_count

    def release(self) -> None:
        """Release OpenCV video capture device."""
        if self._cap.isOpened():
            self._cap.release()