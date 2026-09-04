"""Main orchestrator for real-time video capture and skeletal snapshot logging."""

import sys
import cv2

from capture.camera_handler import CameraHandler
import config
from output.report_generator import ReportGenerator
from output.visualizer import SkeletonVisualizer
from pose.pose_estimator import YOLOSkeletonEstimator


def main() -> None:
    """Initialize camera, run pose model, and log snapshots."""
    print("==================================================================")
    print("        REAL-TIME SKELETAL POSE CAPTURE SYSTEM (YOLOv8)          ")
    print("==================================================================")

    camera = CameraHandler()
    if not camera.is_opened():
        print(f"[ERROR] Could not open camera {config.CAMERA_INDEX}.")
        sys.exit(1)

    pose_estimator = YOLOSkeletonEstimator("yolov8n-pose.pt")
    visualizer = SkeletonVisualizer()
    reporter = ReportGenerator()

    window_name = "Real-Time Skeletal Pose Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"[INFO] Auto-photo capture every {config.SNAPSHOT_INTERVAL_SECONDS}s.")
    print("[INFO] Press 'S' to force snapshot. Press 'Q' to quit.\n")

    try:
        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                break

            # 1. Detect body keypoints using YOLO
            keypoints = pose_estimator.estimate(frame)

            # 2. Capture snapshot every 6s or on 'S' keypress
            if camera.should_capture():
                snapshot_frame = frame.copy()
                if keypoints:
                    snapshot_frame = visualizer.draw_skeleton(snapshot_frame, keypoints)

                snapshot_frame = visualizer.draw_hud(
                    snapshot_frame,
                    countdown_sec=0.0,
                    is_flash_active=True,
                    is_tracked=keypoints is not None,
                )

                reporter.save_snapshot(
                    frame_bgr=snapshot_frame,
                    snapshot_index=camera.snapshot_index,
                    keypoints=keypoints,
                )

            # 3. Live video display
            display_frame = frame.copy()
            if keypoints:
                display_frame = visualizer.draw_skeleton(display_frame, keypoints)

            display_frame = visualizer.draw_hud(
                display_frame,
                countdown_sec=camera.get_countdown_seconds(),
                is_flash_active=camera.is_in_flash_state(),
                is_tracked=keypoints is not None,
            )

            cv2.imshow(window_name, display_frame)

            # 4. Keyboard hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                camera.trigger_manual_snapshot()

    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up and exited.")


if __name__ == "__main__":
    main()
    
    