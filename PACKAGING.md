# Packaging into a Windows app (.exe)

This turns the project into a standalone folder with `DivineSmartCam.exe` that runs on a Windows PC
**without** installing Python. Build it **on a Windows machine** (PyInstaller produces a native binary
for whatever OS it runs on — you can't build a Windows .exe from a Mac).

## One-time setup on the Windows PC

1. **Install Python 3.11** (64-bit) from python.org — tick "Add Python to PATH".

2. **Clone the repo and enter it**
   ```
   git clone https://github.com/vanshxai/divine-majenta-smart-cam.git
   cd divine-majenta-smart-cam
   ```

3. **Create a virtual environment and install deps**
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (Torch will pull the CPU build automatically on a machine with no NVIDIA GPU.)

4. **Get the three model files** (gitignored — not in the repo). Put them in the repo root:
   - `yunet.onnx`  — OpenCV Zoo face_detection_yunet
   - `sface.onnx`  — OpenCV Zoo face_recognition_sface
   - `yolov8n.pt`  — Ultralytics (also auto-downloads on first `python app.py` run; easiest is to just run
     the app once to fetch it, then stop it)

5. **Get ffmpeg for Windows** (ffmpeg.exe + ffprobe.exe from ffmpeg.org). Either put them on the system
   PATH, or drop both into a folder named `ffmpeg` in the repo root.

## Build

```
build_windows.bat
```
This checks the model/ffmpeg files, installs PyInstaller, and builds. Result:
```
dist\DivineSmartCam\DivineSmartCam.exe
```

## Configure and run

Put these files **next to the built .exe** (in `dist\DivineSmartCam\`):

- `config.json` — copy `config.example.json`, set `camera_ip` to your camera's IP, keep `use_proxy` false.
- `.camera_user` — the camera username (one line, e.g. `admin`)
- `.camera_pw` — the camera password (one line)
- an `ffmpeg\` folder with `ffmpeg.exe` + `ffprobe.exe` (unless ffmpeg is on PATH)

Then double-click `DivineSmartCam.exe` and open **http://localhost:5055** in a browser.

The app creates `events.db` and a `snapshots\` folder next to the exe on first run. Attendance windows
are set in the UI (🕐 Attendance) — nothing is recorded outside those windows.

## Notes

- **One folder, not one file.** The build is a folder (faster, more reliable with torch than a single
  giant .exe). Zip the whole `DivineSmartCam` folder to move it to another PC.
- **Antivirus** sometimes flags fresh PyInstaller exes — a known false positive; allow it if needed.
- **Size** ~1–2 GB because torch is bundled. Normal for this stack.
- To make the window silent (no console), set `console=False` in `camera_ai.spec` and rebuild.
