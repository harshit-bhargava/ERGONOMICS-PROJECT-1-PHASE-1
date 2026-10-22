"""Save analyzed frames and their JSON pose reports."""

from datetime import datetime
import html
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from pose.keypoint_schema import KeypointData
from analysis.rula_scoring import _TABLE_A, _TABLE_B, _TABLE_C
import config


class ReportGenerator:
    """Save snapshot images and JSON reports."""

    def __init__(self, output_dir: Path = config.BASE_OUTPUT_DIR) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        frame_bgr: np.ndarray,
        snapshot_index: int,
        keypoints: Optional[Dict[int, KeypointData]],
        joint_angles: Dict[str, float],
        source_type: str = "live",
        source_name: Optional[str] = None,
        source_time_seconds: Optional[float] = None,
    ) -> Tuple[Path, Path]:
        """Save a frame image and its pose, angle, and source metadata."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        iso_timestamp = now.isoformat()

        img_path = self._output_dir / (
            f"frame_{timestamp_str}_{snapshot_index:04d}.jpg"
        )
        json_path = self._output_dir / (
            f"report_{timestamp_str}_{snapshot_index:04d}.json"
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.save(img_path, format="JPEG", quality=95)

        keypoint_data = {}
        if keypoints:
            for idx, keypoint in keypoints.items():
                if idx in (1, 2, 3, 4):
                    continue

                keypoint_data[keypoint.name] = {
                    "id": keypoint.id,
                    "x": round(keypoint.x, 4),
                    "y": round(keypoint.y, 4),
                    "confidence": round(keypoint.visibility, 4),
                }

        payload = {
            "timestamp": iso_timestamp,
            "snapshot_index": snapshot_index,
            "image_file": img_path.name,
            "source_type": source_type,
            "source_name": source_name,
            "source_time_seconds": source_time_seconds,
            "body_detected": keypoints is not None,
            "joint_angles": joint_angles,
            "keypoints": keypoint_data,
        }

        with open(json_path, "w", encoding="utf-8") as report_file:
            json.dump(payload, report_file, indent=2)

        print(
            f"[INFO] Snapshot #{snapshot_index} saved "
            f"with {len(joint_angles)} computed angles: {json_path}"
        )
        return img_path, json_path

    def generate_html_summary(
        self,
        report_paths: list[Path],
        source_type: str,
        source_name: Optional[str] = None,
    ) -> Path:
        """Create a readable, self-contained HTML summary for one session."""
        records = []
        for report_path in report_paths:
            with open(report_path, "r", encoding="utf-8") as report_file:
                record = json.load(report_file)
            record.setdefault(
                "image_file",
                report_path.name.replace("report_", "frame_", 1).replace(
                    ".json", ".jpg"
                ),
            )
            records.append(record)

        created_at = datetime.now()
        file_stamp = created_at.strftime("%Y%m%d_%H%M%S")
        safe_source = "video" if source_type == "video" else "live"
        html_path = self._output_dir / f"summary_{safe_source}_{file_stamp}.html"

        angle_labels = {
            "trunk_inclination_from_vertical_deg": "Trunk inclination from vertical",
            "head_to_trunk_relative_deg": "Head-to-trunk relative angle",
            "l_upper_arm_relative_to_trunk_deg": "Left upper arm relative to trunk",
            "r_upper_arm_relative_to_trunk_deg": "Right upper arm relative to trunk",
            "l_elbow_interior_deg": "Left elbow interior angle",
            "r_elbow_interior_deg": "Right elbow interior angle",
            "l_elbow_flexion_deg": "Left elbow flexion",
            "r_elbow_flexion_deg": "Right elbow flexion",
            "l_knee_interior_deg": "Left knee interior angle",
            "r_knee_interior_deg": "Right knee interior angle",
            "l_hip_interior_deg": "Left hip interior angle",
            "r_hip_interior_deg": "Right hip interior angle",
        }

        def esc(value) -> str:
            return html.escape(str(value))

        def format_angle(value) -> str:
            if isinstance(value, (int, float)):
                return f"{value:.1f}°"
            return "Not available"

        def format_video_time(seconds) -> str:
            if not isinstance(seconds, (int, float)):
                return "—"
            total_seconds = int(seconds)
            minutes, remaining_seconds = divmod(total_seconds, 60)
            return f"{minutes:02d}:{remaining_seconds:02d}"

        def pose_overlay(record: dict, image_file: str) -> str:
            """Build an SVG skeleton and angle-label overlay for a saved frame."""
            keypoints = record.get("keypoints") or {}
            try:
                with Image.open(self._output_dir / image_file) as sample_image:
                    image_width, image_height = sample_image.size
            except (OSError, TypeError):
                return ""

            points = {}
            for point in keypoints.values():
                if not isinstance(point, dict):
                    continue
                point_id = point.get("id")
                confidence = point.get("confidence", 0)
                if (
                    isinstance(point_id, int)
                    and isinstance(point.get("x"), (int, float))
                    and isinstance(point.get("y"), (int, float))
                    and isinstance(confidence, (int, float))
                    and confidence >= 0.25
                ):
                    points[point_id] = (
                        float(point["x"]) * image_width,
                        float(point["y"]) * image_height,
                    )

            connections = (
                (0, 17), (17, 5), (17, 6), (5, 6),
                (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16),
            )
            line_width = max(2.0, image_width / 260)
            point_radius = max(3.5, image_width / 145)
            font_size = max(14.0, image_width / 38)
            svg_parts = [
                f'<svg class="pose-overlay" viewBox="0 0 {image_width} {image_height}" '
                'preserveAspectRatio="none" aria-label="Tracked skeleton and joint angles">'
            ]

            for start_id, end_id in connections:
                if start_id in points and end_id in points:
                    x1, y1 = points[start_id]
                    x2, y2 = points[end_id]
                    svg_parts.append(
                        f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                        f'x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="#00d4ff" stroke-width="{line_width:.1f}" '
                        'stroke-linecap="round" opacity="0.95" />'
                    )

            for point_id, (x, y) in points.items():
                svg_parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{point_radius:.1f}" '
                    'fill="#ff9d00" stroke="#111827" stroke-width="2" />'
                )

            label_positions = {
                "trunk_inclination_from_vertical_deg": (None, -18, -12),
                "l_upper_arm_relative_to_trunk_deg": (5, -12, -14),
                "r_upper_arm_relative_to_trunk_deg": (6, 12, -14),
                "l_elbow_flexion_deg": (7, -12, -12),
                "r_elbow_flexion_deg": (8, 12, -12),
                "l_knee_interior_deg": (13, -12, -12),
                "r_knee_interior_deg": (14, 12, -12),
                "l_hip_interior_deg": (11, -12, 22),
                "r_hip_interior_deg": (12, 12, 22),
                "l_wrist_deviation_deg": (9, -12, -12),
                "r_wrist_deviation_deg": (10, 12, -12),
            }
            angles = record.get("joint_angles") or {}
            for angle_name, (point_id, offset_x, offset_y) in label_positions.items():
                angle_value = angles.get(angle_name)
                if not isinstance(angle_value, (int, float)):
                    continue
                if point_id is None:
                    anchor_ids = (5, 6, 11, 12)
                    if not all(anchor_id in points for anchor_id in anchor_ids):
                        continue
                    shoulder_mid = (
                        sum(points[anchor_id][0] for anchor_id in (5, 6)) / 2,
                        sum(points[anchor_id][1] for anchor_id in (5, 6)) / 2,
                    )
                    hip_mid = (
                        sum(points[anchor_id][0] for anchor_id in (11, 12)) / 2,
                        sum(points[anchor_id][1] for anchor_id in (11, 12)) / 2,
                    )
                    x = (shoulder_mid[0] + hip_mid[0]) / 2
                    y = (shoulder_mid[1] + hip_mid[1]) / 2
                elif point_id in points:
                    x, y = points[point_id]
                else:
                    continue
                label = f"{angle_value:.0f}°"
                svg_parts.append(
                    f'<text x="{x + offset_x:.1f}" y="{y + offset_y:.1f}" '
                    f'font-size="{font_size:.1f}" font-weight="700" '
                    'font-family="Arial, sans-serif" text-anchor="middle" '
                    'fill="#ffffff" stroke="#111827" stroke-width="4" '
                    'paint-order="stroke" stroke-linejoin="round">'
                    f'{label}</text>'
                )

            svg_parts.append("</svg>")
            return "".join(svg_parts)

        detected_count = sum(bool(item.get("body_detected")) for item in records)
        session_title = (
            f"Video analysis: {source_name}"
            if source_type == "video" and source_name
            else "Live camera analysis"
        )

        rows = []
        sample_options = []
        for record in records:
            angles = record.get("joint_angles") or {}
            timestamp = record.get("source_time_seconds")
            readable_time = (
                format_video_time(timestamp)
                if source_type == "video"
                else esc(record.get("timestamp", "—"))
            )
            sample_options.append(
                f'<option value="{record.get("snapshot_index", "")}" '
                f'data-time="{esc(timestamp if timestamp is not None else "")}">'
                f'Sample {record.get("snapshot_index", "—")} · {readable_time}'
                "</option>"
            )
            status_class = "yes" if record.get("body_detected") else "no"
            status_text = "Detected" if record.get("body_detected") else "Not detected"

            keypoint_items = list((record.get("keypoints") or {}).items())
            confidence_values = [
                item.get("confidence")
                for _name, item in keypoint_items
                if isinstance(item.get("confidence"), (int, float))
            ]
            avg_confidence = (
                f"{sum(confidence_values) / len(confidence_values) * 100:.0f}%"
                if confidence_values
                else "—"
            )

            overview_keys = (
                "trunk_inclination_from_vertical_deg",
                "l_upper_arm_relative_to_trunk_deg",
                "r_upper_arm_relative_to_trunk_deg",
                "l_elbow_flexion_deg",
                "r_elbow_flexion_deg",
            )
            overview_cells = "".join(
                f"<td>{format_angle(angles.get(key))}</td>"
                for key in overview_keys
            )

            angle_rows = "".join(
                "<tr>"
                f"<td>{esc(angle_labels.get(key, key.replace('_', ' ').removesuffix('_deg').title()))}</td>"
                f"<td>{format_angle(value)}</td>"
                "</tr>"
                for key, value in angles.items()
            ) or '<tr><td colspan="2">No angle measurements for this frame.</td></tr>'

            keypoint_rows = "".join(
                "<tr>"
                f"<td>{esc(name.replace('_', ' '))}</td>"
                f"<td>{item.get('confidence', 0) * 100:.0f}%</td>"
                "</tr>"
                for name, item in keypoint_items
            ) or '<tr><td colspan="2">No body keypoints were recorded.</td></tr>'

            image_file = record.get("image_file")
            image_html = (
                f'<div class="sample-frame"><img class="sample-image" '
                f'src="{esc(image_file)}" '
                f'alt="Sample frame {record.get("snapshot_index", "")}">'
                f'{pose_overlay(record, image_file)}</div>'
                if image_file
                else ""
            )

            rows.append(
                "<tr>"
                f"<td>{record.get('snapshot_index', '—')}</td>"
                f"<td>{readable_time}</td>"
                f'<td><span class="status {status_class}">{status_text}</span></td>'
                f"{overview_cells}"
                f"<td>{avg_confidence}</td>"
                "</tr>"
                "<tr class=\"details-row\"><td colspan=\"9\"><details>"
                "<summary>View sample frame and detailed measurements</summary>"
                f"{image_html}"
                "<div class=\"detail-grid\"><section><h3>Joint angles</h3>"
                f"<table><tbody>{angle_rows}</tbody></table></section>"
                "<section><h3>Tracked keypoints</h3>"
                f"<table><tbody>{keypoint_rows}</tbody></table></section></div>"
                "</details></td></tr>"
            )

        def score_options(name: str, label: str, maximum: int) -> str:
            options = '<option value="">Choose a score</option>'
            options += "".join(
                f'<option value="{score}">{score}</option>'
                for score in range(1, maximum + 1)
            )
            return (
                f'<label>{esc(label)}<select name="{name}" required>'
                f"{options}</select></label>"
            )

        def checkbox_option(name: str, label: str) -> str:
            return (
                f'<label class="check-option"><input type="checkbox" '
                f'name="{name}" value="yes"><span>{esc(label)}</span></label>'
            )

        def automatic_score_options(name: str, label: str, maximum: int) -> str:
            options = '<option value="">Automatic score unavailable</option>'
            options += "".join(
                f'<option value="{score}">Use manual score {score}</option>'
                for score in range(1, maximum + 1)
            )
            return (
                f'<label>{esc(label)}<select name="{name}" required>'
                f"{options}</select></label>"
            )

        category_fields = "".join(
            (
                automatic_score_options("upper_arm_score", "Upper-arm category", 4),
                automatic_score_options("lower_arm_score", "Lower-arm category", 2),
                automatic_score_options("wrist_score", "Wrist category", 3),
                score_options("wrist_twist_score", "Wrist-twist score (1–2)", 2),
                score_options("neck_score", "Neck score (1–4)", 4),
                automatic_score_options("trunk_score", "Trunk category", 4),
            )
        )
        modifier_fields = f"""
        <div class="modifier-group"><h4>Upper arm</h4>
          {checkbox_option("shoulder_raised", "Shoulder raised (+1)")}
          {checkbox_option("upper_arm_abducted", "Upper arm abducted (+1)")}
          {checkbox_option("arm_supported_or_person_leaning", "Arm supported or leaning (-1)")}
        </div>
        <div class="modifier-group"><h4>Lower arm and wrist</h4>
          {checkbox_option("arm_across_midline_or_out_to_side", "Arm across midline or out to side (+1)")}
          {checkbox_option("wrist_bent_from_midline", "Wrist bent away from midline (+1)")}
        </div>
        <div class="modifier-group"><h4>Neck and trunk</h4>
          {checkbox_option("neck_twisted", "Neck twisted (+1)")}
          {checkbox_option("neck_side_bent", "Neck side-bent (+1)")}
          {checkbox_option("trunk_twisted", "Trunk twisted (+1)")}
          {checkbox_option("trunk_side_bent", "Trunk side-bent (+1)")}
        </div>
        <div class="modifier-group"><h4>Legs</h4>
          {checkbox_option("legs_and_feet_unsupported", "Legs/feet unsupported or not evenly balanced (score 2)")}
        </div>
        """
        assessment_form = f"""<section class="section rula-section">
    <h2>RULA assessment</h2>
    <details>
      <summary>Enter assessment details</summary>
      <form id="rula-form">
        <div class="form-grid">
          <label>Sample posture to score<select name="sample_index" required>
            <option value="">Choose a sample</option>{''.join(sample_options)}
          </select></label>
          <label>Assessed side<select name="side" required>
            <option value="">Choose a side</option><option value="right">Right</option><option value="left">Left</option>
          </select></label>
          <label>Camera view<select name="camera_view" required>
            <option value="">Choose the worker's view in the image</option>
            <option value="side_facing_left">Side-on; worker faces left</option>
            <option value="side_facing_right">Side-on; worker faces right</option>
            <option value="front">Front or back view</option>
            <option value="oblique">Oblique or unclear view</option>
          </select></label>
        </div>
        <h3>Posture categories</h3>
        <div class="form-grid">{category_fields}</div>
        <h3>Posture adjustments</h3>
        <div class="modifier-grid">{modifier_fields}</div>
        <h3>Muscle use and force/load</h3>
        <div class="form-grid">
          <label>Muscle-use score<select name="muscle_use" required>
            <option value="">Choose yes or no</option><option value="yes">Yes: static over 1 minute or repeated over 4 times/minute (+1)</option><option value="no">No (0)</option>
          </select></label>
          <label>Force/load category<select name="force_load_category" required>
            <option value="">Choose a category</option>
            <option value="0:intermittent-under-5">No resistance or intermittent load under 5 lb (0)</option>
            <option value="1:intermittent-5-20">5–20 lb, intermittent (1)</option>
            <option value="2:static-repeated-5-20">5–20 lb, static or repeated (2)</option>
            <option value="3:static-repeated-over-20">Over 20 lb, static or repeated (3)</option>
            <option value="3:shock">Shock or rapidly building force (3)</option>
          </select></label>
        </div>
        <button type="submit" class="calculate-button">Calculate RULA score</button>
      </form>
      <div id="rula-result" class="rula-result" aria-live="polite"></div>
      <button id="download-rula" class="download-button" type="button" hidden>Download RULA assessment JSON</button>
    </details>
  </section>"""

        posture_samples = json.dumps(
            [
                {
                    "snapshot_index": record.get("snapshot_index"),
                    "joint_angles": record.get("joint_angles") or {},
                }
                for record in records
            ]
        )
        rula_script = """
<script>
const TABLE_A = __TABLE_A__;
const TABLE_B = __TABLE_B__;
const TABLE_C = __TABLE_C__;
const SAMPLES = __SAMPLES__;
const form = document.getElementById("rula-form");
const resultBox = document.getElementById("rula-result");
const downloadButton = document.getElementById("download-rula");
let assessmentForDownload = null;

function setAutomaticScore(name, score, measurementText) {
  const select = form.elements[name];
  const automaticOption = select.options[0];
  if (score === null || !Number.isFinite(score)) {
    automaticOption.value = "";
    automaticOption.textContent = "Automatic score unavailable";
  } else {
    automaticOption.value = `auto:${score}`;
    automaticOption.textContent = `Automatic estimate: ${score} (${measurementText})`;
  }
  select.selectedIndex = 0;
}

function estimateTrunkScore(angle, cameraView) {
  if (!Number.isFinite(angle) ||
      !["side_facing_left", "side_facing_right"].includes(cameraView)) {
    return { score: null, detail: "side-on angle unavailable" };
  }

  // The sign of the image-plane angle depends on which way the worker faces.
  const forwardSignedAngle = cameraView === "side_facing_right" ? angle : -angle;
  const forwardBend = Math.max(0, forwardSignedAngle);
  const extension = forwardSignedAngle < 0;
  let score;
  if (extension) score = 2;
  else if (forwardBend < 5) score = 1;
  else if (forwardBend <= 20) score = 2;
  else if (forwardBend <= 60) score = 3;
  else score = 4;

  return {
    score,
    detail: `${angle.toFixed(1)}° image angle; ${extension ? "extension" : `${forwardBend.toFixed(1)}° forward bend`}`
  };
}

function refreshAutomaticTrunkScore() {
  const sampleIndex = Number(form.elements.sample_index.value);
  const sample = SAMPLES.find(item => item.snapshot_index === sampleIndex);
  const angles = sample ? sample.joint_angles : {};
  const estimate = estimateTrunkScore(
    angles.trunk_inclination_from_vertical_deg,
    form.elements.camera_view.value
  );
  setAutomaticScore("trunk_score", estimate.score, estimate.detail);
}

function refreshAutomaticScores() {
  const sampleIndex = Number(form.elements.sample_index.value);
  const selectedSide = form.elements.side.value;
  const sample = SAMPLES.find(item => item.snapshot_index === sampleIndex);
  const angles = sample ? sample.joint_angles : {};
  const prefix = selectedSide === "left" ? "l" : "r";
  let upperArmScore = null;
  let lowerArmScore = null;
  let wristScore = null;

  if (selectedSide) {
    const armAngle = angles[`${prefix}_upper_arm_relative_to_trunk_deg`];
    if (Number.isFinite(armAngle)) {
      const armAngleMagnitude = Math.abs(armAngle);
      if (armAngleMagnitude <= 20) upperArmScore = 1;
      else if (armAngleMagnitude <= 45) upperArmScore = 2;
      else if (armAngleMagnitude <= 90) upperArmScore = 3;
      else upperArmScore = 4;
    }

    const elbowFlexion = angles[`${prefix}_elbow_flexion_deg`];
    if (Number.isFinite(elbowFlexion)) {
      lowerArmScore = elbowFlexion >= 60 && elbowFlexion <= 100 ? 1 : 2;
    }

    const wristDeviation = angles[`${prefix}_wrist_deviation_deg`];
    if (Number.isFinite(wristDeviation)) {
      const absoluteWristAngle = Math.abs(wristDeviation);
      wristScore = absoluteWristAngle === 0 ? 1 :
        (absoluteWristAngle <= 15 ? 2 : 3);
    }

  }

  setAutomaticScore(
    "upper_arm_score", upperArmScore,
    Number.isFinite(angles[`${prefix}_upper_arm_relative_to_trunk_deg`])
      ? `${Math.abs(angles[`${prefix}_upper_arm_relative_to_trunk_deg`]).toFixed(1)}° from neutral`
      : "angle unavailable"
  );
  setAutomaticScore(
    "lower_arm_score", lowerArmScore,
    Number.isFinite(angles[`${prefix}_elbow_flexion_deg`])
      ? `${angles[`${prefix}_elbow_flexion_deg`].toFixed(1)}° elbow flexion`
      : "angle unavailable"
  );
  setAutomaticScore(
    "wrist_score", wristScore,
    Number.isFinite(angles[`${prefix}_wrist_deviation_deg`])
      ? `${Math.abs(angles[`${prefix}_wrist_deviation_deg`]).toFixed(1)}° wrist angle`
      : "hand angle unavailable"
  );
}

form.addEventListener("change", (event) => {
  if (["sample_index", "side"].includes(event.target.name)) {
    refreshAutomaticScores();
    refreshAutomaticTrunkScore();
  } else if (event.target.name === "camera_view") {
    refreshAutomaticTrunkScore();
  }
});
refreshAutomaticScores();
refreshAutomaticTrunkScore();

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const values = new FormData(form);
  const number = (name) => Number(values.get(name));
  const yes = (name) => values.get(name) === "yes";
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const componentScore = (name) => {
    const value = values.get(name);
    return value && value.startsWith("auto:")
      ? Number(value.slice(5)) : Number(value);
  };
  const baseUpperArm = componentScore("upper_arm_score");
  const baseLowerArm = componentScore("lower_arm_score");
  const baseWrist = componentScore("wrist_score");
  const baseTrunk = componentScore("trunk_score");
  if (![baseUpperArm, baseLowerArm, baseWrist, baseTrunk].every(Number.isFinite)) {
    resultBox.textContent = "Select a manual category for each unavailable automatic estimate.";
    downloadButton.hidden = true;
    return;
  }

  const upperArm = clamp(
    baseUpperArm + Number(yes("shoulder_raised")) +
      Number(yes("upper_arm_abducted")) -
      Number(yes("arm_supported_or_person_leaning")), 1, 6);
  const lowerArm = clamp(
    baseLowerArm +
      Number(yes("arm_across_midline_or_out_to_side")), 1, 3);
  const wrist = clamp(
    baseWrist + Number(yes("wrist_bent_from_midline")), 1, 4);
  const wristTwist = number("wrist_twist_score");
  const neck = clamp(
    number("neck_score") + Number(yes("neck_twisted")) +
      Number(yes("neck_side_bent")), 1, 6);
  const trunk = clamp(
    baseTrunk + Number(yes("trunk_twisted")) +
      Number(yes("trunk_side_bent")), 1, 6);
  const legs = yes("legs_and_feet_unsupported") ? 2 : 1;

  const tableAIndex = (wrist - 1) * 2 + (wristTwist - 1);
  const postureA = TABLE_A[upperArm - 1][lowerArm - 1][tableAIndex];
  const tableBIndex = (trunk - 1) * 2 + (legs - 1);
  const postureB = TABLE_B[neck - 1][tableBIndex];

  const muscleUse = Number(values.get("muscle_use") === "yes");
  const forceLoadCategory = values.get("force_load_category");
  const forceLoad = Number(forceLoadCategory.split(":")[0]);

  const finalA = postureA + muscleUse + forceLoad;
  const finalB = postureB + muscleUse + forceLoad;
  const finalScore = TABLE_C[Math.min(finalA, 8) - 1][Math.min(finalB, 7) - 1];
  let rating;
  let action;
  let actionLevel;
  let riskLevel;
  if (finalScore <= 2) {
    actionLevel = 1;
    riskLevel = "Negligible";
    rating = "Acceptable";
    action = "Acceptable if the posture is not maintained or repeated for long periods.";
  } else if (finalScore <= 4) {
    actionLevel = 2;
    riskLevel = "Low";
    rating = "Investigate further";
    action = "Investigate further.";
  } else if (finalScore <= 6) {
    actionLevel = 3;
    riskLevel = "Medium";
    rating = "Investigate further and change soon";
    action = "Investigate further and change soon.";
  } else {
    actionLevel = 4;
    riskLevel = "High";
    rating = "Investigate and change immediately";
    action = "Investigate and change immediately.";
  }

  const selectedSample = form.querySelector('[name="sample_index"]');
  const selectedOption = selectedSample.options[selectedSample.selectedIndex];
  const componentScores = {
    upper_arm: upperArm,
    lower_arm: lowerArm,
    wrist: wrist,
    wrist_twist: wristTwist,
    neck: neck,
    trunk: trunk,
    legs: legs
  };
  const observation = Object.fromEntries(values.entries());
  assessmentForDownload = {
    sample_index: Number(values.get("sample_index")),
    sample_label: selectedOption.textContent,
    source_time_seconds: selectedOption.dataset.time
      ? Number(selectedOption.dataset.time) : null,
    observation: observation,
    result: {
      side: values.get("side"),
      adjusted_component_scores: componentScores,
      muscle_use_score: muscleUse,
      force_load_score: forceLoad,
      posture_score_a: postureA,
      posture_score_b: postureB,
      final_wrist_arm_score: finalA,
      final_neck_trunk_leg_score: finalB,
      final_rula_score: finalScore,
      action_level: actionLevel,
      risk_level: riskLevel,
      rating: rating,
      action: action
    }
  };

  resultBox.innerHTML = "";
  const heading = document.createElement("h3");
  heading.textContent = `Final RULA score: ${finalScore}`;
  const ratingLine = document.createElement("p");
  ratingLine.textContent = `Risk level: ${riskLevel} · Action Level ${actionLevel} · Assessed side: ${values.get("side")}`;
  const actionLine = document.createElement("p");
  actionLine.textContent = `Recommended action: ${action}`;
  const breakdown = document.createElement("p");
  breakdown.textContent =
    `Group A: ${postureA} + muscle use ${muscleUse} + force/load ${forceLoad} = ${finalA}. ` +
    `Group B: ${postureB} + muscle use ${muscleUse} + force/load ${forceLoad} = ${finalB}.`;
  resultBox.append(heading, ratingLine, actionLine, breakdown);
  downloadButton.hidden = false;
});

downloadButton.addEventListener("click", () => {
  if (!assessmentForDownload) return;
  const blob = new Blob(
    [JSON.stringify(assessmentForDownload, null, 2)],
    { type: "application/json" }
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `rula_assessment_sample_${assessmentForDownload.sample_index}.json`;
  link.click();
  URL.revokeObjectURL(url);
});
</script>
"""
        rula_script = (
            rula_script.replace("__TABLE_A__", json.dumps(_TABLE_A))
            .replace("__TABLE_B__", json.dumps(_TABLE_B))
            .replace("__TABLE_C__", json.dumps(_TABLE_C))
            .replace("__SAMPLES__", posture_samples)
        )

        source_label = esc(source_name or "Camera")
        report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(session_title)} - Analysis Summary</title>
  <style>
    :root {{ color-scheme: light; --ink:#1f2937; --muted:#64748b; --line:#e2e8f0; --accent:#146c43; --soft:#f4f8f6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#f1f5f3; color:var(--ink); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1250px; margin:36px auto; padding:0 22px 48px; }}
    header {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:26px 30px; box-shadow:0 6px 24px #0f172a0a; }}
    h1 {{ margin:0 0 5px; font-size:27px; }}
    .subtitle {{ color:var(--muted); margin:0; }}
    .cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:18px 0; }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:17px 20px; }}
    .card-label {{ display:block; color:var(--muted); font-size:13px; }}
    .card-value {{ display:block; font-size:22px; font-weight:700; margin-top:4px; overflow-wrap:anywhere; }}
    .section {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:22px; overflow:hidden; }}
    h2 {{ margin:0 0 14px; font-size:19px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:11px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ background:var(--soft); color:#334155; font-weight:650; }}
    .status {{ display:inline-block; border-radius:99px; padding:3px 9px; font-size:12px; font-weight:650; }}
    .yes {{ background:#dcfce7; color:#166534; }} .no {{ background:#fee2e2; color:#991b1b; }}
    .details-row>td {{ padding:0; white-space:normal; }}
    details {{ padding:10px 12px; }} summary {{ color:var(--accent); cursor:pointer; font-weight:650; }}
    .sample-frame {{ position:relative; width:min(100%,520px); margin:16px 0; }}
    .sample-image {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }}
    .pose-overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
    .detail-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }}
    .detail-grid h3 {{ margin:8px 0; font-size:15px; }}
    .note {{ margin:16px 2px 0; color:var(--muted); font-size:13px; }}
    .rula-section {{ margin-top:18px; }}
    .rula-section details {{ padding:0; }}
    .rula-section details>summary {{ padding:12px 0; }}
    .form-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:12px 0 20px; }}
    .form-grid label {{ display:flex; flex-direction:column; gap:6px; color:#334155; font-size:13px; font-weight:600; }}
    .form-grid select,.form-grid input {{ width:100%; min-height:40px; padding:8px 10px; border:1px solid #cbd5e1; border-radius:7px; background:#fff; color:var(--ink); font:inherit; }}
    .modifier-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:12px 0 20px; }}
    .modifier-group {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; background:#fff; }}
    .modifier-group h4 {{ margin:0 0 8px; color:#334155; font-size:14px; }}
    .check-option {{ display:flex; align-items:flex-start; gap:8px; margin:7px 0; color:#334155; font-size:13px; }}
    .check-option input {{ margin:3px 0 0; accent-color:var(--accent); }}
    .rula-section h3 {{ margin:20px 0 8px; font-size:15px; }}
    .calculate-button,.download-button {{ border:0; border-radius:8px; padding:11px 16px; background:var(--accent); color:#fff; font:inherit; font-weight:650; cursor:pointer; }}
    .download-button {{ margin-top:14px; background:#334155; }}
    .rula-result:not(:empty) {{ margin-top:18px; padding:16px; border:1px solid #bbf7d0; border-radius:10px; background:#f0fdf4; }}
    .rula-result h3 {{ margin:0 0 6px; font-size:21px; color:#166534; }}
    .rula-result p {{ margin:6px 0; }}
    footer {{ color:var(--muted); font-size:12px; margin-top:18px; }}
    @media(max-width:760px) {{ main {{ margin:16px auto; padding:0 12px 30px; }} header,.section {{ padding:18px; }} .cards,.form-grid,.modifier-grid {{ grid-template-columns:1fr; }} .detail-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{esc(session_title)}</h1>
    <p class="subtitle">Sampled pose and joint-angle measurements from this analysis session.</p>
  </header>
  <section class="cards" aria-label="Session overview">
    <div class="card"><span class="card-label">Input</span><span class="card-value">{esc(source_type.title())}</span></div>
    <div class="card"><span class="card-label">Source</span><span class="card-value">{source_label}</span></div>
    <div class="card"><span class="card-label">Samples with body detected</span><span class="card-value">{detected_count} / {len(records)}</span></div>
  </section>
  <section class="section">
    <h2>Sample overview</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>{'Video time' if source_type == 'video' else 'Captured at'}</th><th>Tracking</th><th>Trunk angle</th><th>Left upper arm</th><th>Right upper arm</th><th>Left elbow flexion</th><th>Right elbow flexion</th><th>Avg. keypoint confidence</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
  </section>
  {assessment_form}
  <footer>Generated {created_at.strftime('%Y-%m-%d %H:%M:%S')} · Detailed source data remains available in the JSON reports beside this page.</footer>
</main>
{rula_script}
</body>
</html>
"""

        html_path.write_text(report_html, encoding="utf-8")
        return html_path
