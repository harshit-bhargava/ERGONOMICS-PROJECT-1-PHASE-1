"""Camera handler managing video capture, timing intervals, and snapshot triggers."""

import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import config


class CameraHandler:
    """Handles video stream input, countdown timers, and periodic snapshot triggers."""

    def __init__(self, camera_source: Union[int, str] = config.CAMERA_INDEX) -> None:
        self.cap = cv2.VideoCapture(camera_source)
        if isinstance(camera_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

        self.snapshot_index = 0
        self.last_snapshot_time = time.time()
        self.flash_start_time = 0.0
        self.force_trigger = False

    @property
    def is_open(self) -> bool:
        return self.cap.isOpened()

    @property
    def is_flash_active(self) -> bool:
        return (time.time() - self.flash_start_time) < config.FLASH_BANNER_DURATION_SECONDS

    def trigger_flash(self) -> None:
        self.flash_start_time = time.time()

    def force_snapshot(self) -> None:
        self.force_trigger = True

    def read_frame(self) -> Tuple[Optional[np.ndarray], float, bool]:
        """Read frame and return (frame, elapsed_time, trigger_snapshot)."""
        ret, frame = self.cap.read()
        if not ret:
            return None, 0.0, False

        current_time = time.time()
        elapsed_time = current_time - self.last_snapshot_time

        trigger = False
        interval = getattr(config, "SNAPSHOT_INTERVAL_SECONDS", 6.0)

        # Trigger snapshot when interval expires or when manual 'S' key is pressed
        if elapsed_time >= interval or self.force_trigger:
            trigger = True
            self.snapshot_index += 1
            self.last_snapshot_time = current_time
            self.force_trigger = False
            elapsed_time = 0.0

        return frame, elapsed_time, trigger

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()