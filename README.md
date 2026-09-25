# Divine Smart Cam

Offline human-detection + face-recognition + **attendance** system for IP cameras (built against a
Dahua DH-IPC-HDBW3241RP-ZAS), pulling the raw RTSP feed directly and running YOLOv8 + YuNet/SFace
locally — no vendor cloud/AI.

**To build the installable Windows app, see [PACKAGING.md](PACKAGING.md).** The steps below are for
running from source (development).

## Run from source

1. **Python 3.11 env** (needed for a working torch wheel)
   ```
   python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **ffmpeg** — required (opencv-python has no FFmpeg support, so the app pipes frames from the real
   ffmpeg binary). Install it and either put it on PATH, drop `ffmpeg`/`ffprobe` in an `ffmpeg/` folder
   next to `app.py`, or set `ffmpeg_path`/`ffprobe_path` in `config.json`. The app auto-detects PATH.

3. **Face models** — not committed (binary). Download from the OpenCV Zoo into the project root:
   - face_detection_yunet → `yunet.onnx`
   - face_recognition_sface → `sface.onnx`

4. **YOLO weights** — `yolov8n.pt` auto-downloads on first run (needs internet once).

5. **Camera credentials** — two plain-text files in the project root (gitignored):
   ```
   echo -n "admin" > .camera_user
   echo -n "yourpassword" > .camera_pw
   ```

6. **config.json** — copy `config.example.json` → `config.json` and set `camera_ip`. On a normal
   single-network machine keep `use_proxy` false. (The proxy is a Mac-only dual-subnet workaround.)

7. **Run**
   ```
   python -u app.py
   ```
   Opens on `http://localhost:5055`.

## Attendance

Set the morning-in and evening-out windows in the UI (🕐 Attendance). Recognition and the live view
run all day, but **nothing is written to disk/DB outside those windows** — no event rows, no snapshots.
Inside a window, each known person is recorded once: first sighting = check-in (morning), last sighting
= check-out (evening). Records show in the Attendance panel and on the `/database` page.

## Data (not in this repo)

`events.db`, `snapshots/`, `config.json`, and the credential files hold this install's private/biometric
data and are gitignored. Move them separately (USB/scp) only if you want to carry history to another box.

## Layout

Everything — optional proxy, database layer, camera stream + detection pipeline, attendance logic, Flask
routes, and the full frontend (HTML/CSS/JS) — lives in the single `app.py` file.
