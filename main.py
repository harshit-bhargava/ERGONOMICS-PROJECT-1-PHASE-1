"""Master application entry point with real-time pose and joint angle overlays."""

import cv2
import config
from capture.camera_handler import CameraHandler
from pose.pose_estimator import YOLOSkeletonEstimator
from output.visualizer import SkeletonVisualizer
from output.report_generator import ReportGenerator
from analysis.joint_angles import compute_all_joint_angles


def main() -> None:
    camera = CameraHandler()
    estimator = YOLOSkeletonEstimator()
    visualizer = SkeletonVisualizer()
    reporter = ReportGenerator()

    if not camera.is_open:
        print("[ERROR] Camera initialization failed.")
        return

    print("=" * 60)
    print("   REAL-TIME ERGONOMIC POSE & ANGLE TRACKING (YOLOv8)   ")
    print("=" * 60)
    print(f"[INFO] Frame auto-snapshot every {config.SNAPSHOT_INTERVAL_SECONDS}s.")
    print("[INFO] Press 'S' to force snapshot. Press 'Q' to quit.\n")

    try:
        while True:
            frame_raw, elapsed_time, trigger_snapshot = camera.read_frame()
            if frame_raw is None:
                break

            # 1. Skeletal pose estimation (excludes eyes/ears, computes neck)
            keypoints = estimator.estimate(frame_raw)

            # 2. Continuous real-time vector angle calculations
            joint_angles = compute_all_joint_angles(keypoints) if keypoints else {}

            # 3. Handle snapshot trigger (every 6 seconds or manual press)
            if trigger_snapshot:
                reporter.save_snapshot(
                    frame_bgr=frame_raw,
                    snapshot_index=camera.snapshot_index,
                    keypoints=keypoints,
                    joint_angles=joint_angles,
                )
                camera.trigger_flash()

            # 4. Render skeleton lines, joint degree badges, and telemetry card
            display_frame = frame_raw.copy()
            if keypoints:
                display_frame = visualizer.draw_skeleton(display_frame, keypoints)
                display_frame = visualizer.draw_joint_angles(display_frame, keypoints, joint_angles)

            # 5. Render HUD status and countdown timer
            countdown = max(0.0, config.SNAPSHOT_INTERVAL_SECONDS - elapsed_time)
            display_frame = visualizer.draw_hud(
                frame=display_frame,
                countdown_sec=countdown,
                is_flash_active=camera.is_flash_active,
                is_tracked=keypoints is not None,
            )

            # 6. Stream frame to active OpenCV window
            cv2.imshow(config.WINDOW_TITLE, display_frame)

            # 7. Keyboard listener
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                camera.force_snapshot()

    finally:
        camera.release()
        estimator.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Cleaned up and exited.")


if __name__ == "__main__":
    main()