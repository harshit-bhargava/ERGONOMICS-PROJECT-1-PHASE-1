# Real-Time Computer Vision Ergonomics Risk Assessment System

A modular, production-ready Python application that performs live video ergonomic risk evaluations using human skeletal pose estimation and biomechanical vector geometry.

---

## 🌟 Key Features

- **Live Video Pose Estimation**: Extracts 33 3D skeletal landmarks via MediaPipe Pose (with automated fallback to YOLOv8-Pose).
- **Automated 6-Second Sampling**: Captures a high-resolution snapshot and generates a comprehensive risk report every 6 seconds.
- **Biomechanical Vector Calculations**: Formulates real-time planar angles for neck flexion, trunk inclination, shoulder lift, elbow flexion, wrist deviation, hip angle, and knee flexion.
- **Dynamic HUD & Overlays**: Real-time visual feedback with color-coded risk markers (Green/Amber/Red), countdown clocks, and capture confirmation banners.
- **Pluggable Scoring Framework**: Built-in abstract base class architecture ready for **RULA**, **REBA**, and **OCRA** lookup algorithms.
- **Structured Artifact Generation**: Exports timestamped annotated images (`.jpg`) and machine-readable data audits (`.json`).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10 or newer installed:
```bash
python --version# ERGONOMICS-
