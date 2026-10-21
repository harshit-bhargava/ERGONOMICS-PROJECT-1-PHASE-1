# Ergonomics Pose and RULA Assessment

This application analyzes live camera footage or an uploaded video. It estimates body keypoints and joint angles, saves a sample every six seconds, and creates an HTML report with annotated sample frames and an optional RULA assessment form.

## Requirements

- Python 3.12
- Git
- A webcam for live analysis
- macOS or Windows (the video picker uses the native macOS picker or the Windows file dialog)

The YOLO pose model file, `yolov8n-pose.pt`, is included in this repository. The generated `snapshots/` folder contains captured images and reports and is excluded from Git.

## Windows setup

Open **Command Prompt** and run:

```bat
git clone https://github.com/harshit-bhargava/ERGONOMICS-PROJECT-1-PHASE-1.git
cd ERGONOMICS-PROJECT-1-PHASE-1
py -3.12 -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If Windows reports that `py` is not recognized, install Python 3.12 and select the option to add the Python launcher to PATH.

## macOS setup

Open **Terminal** and run:

```bash
git clone https://github.com/harshit-bhargava/ERGONOMICS-PROJECT-1-PHASE-1.git
cd ERGONOMICS-PROJECT-1-PHASE-1
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Install Python 3.12 first if `python3.12` is not available. macOS uses its built-in file picker; Windows uses Python's Tk file dialog.

## Using the application

1. Run `python main.py` from the cloned project folder with the virtual environment activated.
2. Choose **Live Camera** or **Upload Video**.
3. For a video, select a supported video file in the file picker. The app samples it at six-second intervals.
4. When analysis ends, the HTML summary opens in the default browser. Detailed reports and sample images are saved in `snapshots/`.

Press **Q** to stop analysis. In live mode, press **S** to take a snapshot immediately.

## Dependencies

Dependencies are listed in `requirements.txt`. Install them inside the virtual environment with `python -m pip install -r requirements.txt`.
