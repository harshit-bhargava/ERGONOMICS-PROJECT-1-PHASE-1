"""Entry point for live-camera and uploaded-video pose analysis."""

from pathlib import Path
import subprocess
import webbrowser

import cv2
import numpy as np

import config
from analysis.joint_angles import compute_all_joint_angles
from capture.camera_handler import CameraHandler
from output.report_generator import ReportGenerator
from output.visualizer import SkeletonVisualizer
from pose.hand_estimator import MediaPipeHandTracker
from pose.pose_estimator import YOLOSkeletonEstimator


def choose_input_mode() -> tuple[str | None, str | None]:
    """Show clickable source buttons and a native macOS video picker."""
    window_name = "Ergonomics Analysis - Choose Source"
    canvas = np.full((360, 900, 3), 245, dtype=np.uint8)
    selection = {"mode": None, "video_path": None, "open_picker": False}

    def draw_button(image, left, top, right, bottom, label):
        cv2.rectangle(image, (left, top), (right, bottom), (35, 125, 70), -1)
        cv2.rectangle(image, (left, top), (right, bottom), (20, 75, 40), 2)
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )[0]
        text_x = left + (right - left - text_size[0]) // 2
        text_y = top + (bottom - top + text_size[1]) // 2
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def mouse_callback(event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if 100 <= x <= 420 and 170 <= y <= 270:
            selection["mode"] = "live"
        elif 480 <= x <= 800 and 170 <= y <= 270:
            selection["open_picker"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while selection["mode"] is None:
            canvas[:] = (245, 245, 245)
            cv2.putText(
                canvas,
                "Choose an analysis source",
                (215, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (35, 35, 35),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Click an option below to continue",
                (260, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            draw_button(canvas, 100, 170, 420, 270, "1. Live Camera")
            draw_button(canvas, 480, 170, 800, 270, "2. Upload Video")
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break

            if selection["open_picker"]:
                selection["open_picker"] = False
                picker_script = (
                    'POSIX path of (choose file with prompt '
                    '"Select a video to analyze" of type {"public.movie"})'
                )
                try:
                    result = subprocess.run(
                        ["osascript", "-e", picker_script],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except FileNotFoundError:
                    print("[ERROR] macOS osascript was not found.")
                    continue

                if result.returncode == 0 and result.stdout.strip():
                    selection["mode"] = "video"
                    selection["video_path"] = result.stdout.strip()
                # Canceling the picker leaves the two source buttons available.

    finally:
        cv2.destroyWindow(window_name)

    return selection["mode"], selection["video_path"]


def analyze_frame(frame_bgr, estimator, hand_tracker, visualizer):
    """Run pose, hand, angle, and overlay processing on one frame."""
    keypoints = estimator.estimate(frame_bgr)
    hand_keypoints = hand_tracker.process_frame(
        frame_bgr,
        body_keypoints=keypoints,
    )

    joint_angles = (
        compute_all_joint_angles(
            keypoints,
            hand_keypoints=hand_keypoints,
        )
        if keypoints
        else {}
    )

    display_frame = frame_bgr.copy()
    if keypoints:
        display_frame = visualizer.draw_skeleton(
            display_frame,
            keypoints,
            hand_keypoints=hand_keypoints,
        )
        display_frame = visualizer.draw_joint_angles(
            display_frame,
            keypoints,
            joint_angles,
            hand_keypoints=hand_keypoints,
        )

    return keypoints, joint_angles, display_frame


def create_and_open_summary(reporter, report_paths, source_type, source_name=None):
    """Create this session's HTML summary and open it in the default browser."""
    if not report_paths:
        print("[INFO] No snapshots were saved, so no summary page was created.")
        return

    summary_path = reporter.generate_html_summary(
        report_paths=report_paths,
        source_type=source_type,
        source_name=source_name,
    )
    print(f"[INFO] HTML summary saved: {summary_path.resolve()}")

    if not webbrowser.open(summary_path.resolve().as_uri()):
        print("[WARNING] Could not open the summary automatically.")


def run_live_camera(estimator, hand_tracker, visualizer, reporter) -> None:
    """Run the existing live-camera workflow."""
    camera = CameraHandler()

    if not camera.is_open:
        print("[ERROR] Camera initialization failed.")
        camera.release()
        return

    print(
        f"[INFO] Live snapshots are saved every "
        f"{config.SNAPSHOT_INTERVAL_SECONDS} seconds."
    )
    print("[INFO] Press S to save a snapshot. Press Q to quit.")
    report_paths = []

    try:
        while True:
            frame_raw, elapsed_time, trigger_snapshot = camera.read_frame()
            if frame_raw is None:
                print("[ERROR] Could not read a frame from the camera.")
                break

            keypoints, joint_angles, display_frame = analyze_frame(
                frame_raw,
                estimator,
                hand_tracker,
                visualizer,
            )

            if trigger_snapshot:
                _image_path, report_path = reporter.save_snapshot(
                    frame_bgr=frame_raw,
                    snapshot_index=camera.snapshot_index,
                    keypoints=keypoints,
                    joint_angles=joint_angles,
                    source_type="live",
                )
                report_paths.append(report_path)
                camera.trigger_flash()

            countdown = max(
                0.0,
                config.SNAPSHOT_INTERVAL_SECONDS - elapsed_time,
            )
            display_frame = visualizer.draw_hud(
                frame=display_frame,
                countdown_sec=countdown,
                is_flash_active=camera.is_flash_active,
                is_tracked=keypoints is not None,
            )

            cv2.imshow(config.WINDOW_TITLE, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                camera.force_snapshot()

    finally:
        camera.release()
        cv2.destroyAllWindows()

    create_and_open_summary(
        reporter=reporter,
        report_paths=report_paths,
        source_type="live",
    )


def run_video_file(
    video_path_text: str,
    estimator,
    hand_tracker,
    visualizer,
    reporter,
) -> None:
    """Analyze and save a frame every configured number of video seconds."""
    video_path = Path(video_path_text).expanduser()
    if not video_path.is_file():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        video.release()
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("[ERROR] Could not read the video's frame rate.")
        video.release()
        return

    interval_seconds = float(config.SNAPSHOT_INTERVAL_SECONDS)
    if interval_seconds <= 0:
        print("[ERROR] Snapshot interval must be greater than zero.")
        video.release()
        return

    # Sampling begins at frame 0, then advances by six seconds of video time.
    interval_frames = max(1, round(fps * interval_seconds))
    frame_index = 0
    snapshot_index = 0
    next_sample_frame = 0
    report_paths = []

    print(f"[INFO] Analyzing video: {video_path.name}")
    print(f"[INFO] Video frame rate: {fps:.2f} FPS")
    print(
        f"[INFO] Saving one sample every {interval_seconds:g} "
        "video seconds, starting at 0 seconds."
    )
    print("[INFO] Press Q in the video window to stop early.")

    try:
        while True:
            success, frame_raw = video.read()
            if not success:
                print("[INFO] Reached the end of the video.")
                break

            if frame_index >= next_sample_frame:
                source_time_seconds = frame_index / fps
                keypoints, joint_angles, display_frame = analyze_frame(
                    frame_raw,
                    estimator,
                    hand_tracker,
                    visualizer,
                )

                snapshot_index += 1
                _image_path, report_path = reporter.save_snapshot(
                    frame_bgr=frame_raw,
                    snapshot_index=snapshot_index,
                    keypoints=keypoints,
                    joint_angles=joint_angles,
                    source_type="video",
                    source_name=video_path.name,
                    source_time_seconds=round(source_time_seconds, 3),
                )
                report_paths.append(report_path)

                next_sample_frame += interval_frames
                cv2.imshow(config.WINDOW_TITLE, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Video analysis stopped by user.")
                    break

            frame_index += 1

    finally:
        video.release()
        cv2.destroyAllWindows()

    create_and_open_summary(
        reporter=reporter,
        report_paths=report_paths,
        source_type="video",
        source_name=video_path.name,
    )


def main() -> None:
    mode, video_path = choose_input_mode()
    if mode is None:
        print("[INFO] No analysis mode was selected.")
        return

    estimator = YOLOSkeletonEstimator()
    hand_tracker = MediaPipeHandTracker()
    visualizer = SkeletonVisualizer()
    reporter = ReportGenerator()

    try:
        if mode == "live":
            run_live_camera(
                estimator,
                hand_tracker,
                visualizer,
                reporter,
            )
        elif mode == "video" and video_path is not None:
            run_video_file(
                video_path,
                estimator,
                hand_tracker,
                visualizer,
                reporter,
            )
    finally:
        estimator.release()
        hand_tracker.release()
        cv2.destroyAllWindows()

    print("[INFO] Analysis finished.")


if __name__ == "__main__":
    main()
