# camera_ai

Offline human-detection + face-recognition system for a Dahua IP camera (DH-IPC-HDBW3241RP-ZAS),
pulling the raw RTSP feed directly and running YOLOv8 + YuNet/SFace locally — no vendor cloud/AI.

## Setup on a new machine

1. **Conda env (Python 3.11 — needed for a working Intel/Apple torch wheel)**
   ```
   conda create -n camera_ai_py311 python=3.11 -y
   conda activate camera_ai_py311
   pip install -r requirements.txt
   ```

2. **ffmpeg** — required; the app pipes raw frames from the real `ffmpeg` binary
   (opencv-python's own build has no FFmpeg support). Install via conda or your OS package
   manager, then update `FFMPEG_BIN` / `FFPROBE_BIN` near the top of `app.py` to match its path
   (`which ffmpeg`).

3. **Face models** — not committed (binary, ~39MB combined). Download from the OpenCV Zoo:
   - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet → save as `yunet.onnx`
   - https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface → save as `sface.onnx`

   Both go in the project root, alongside `app.py`.

4. **YOLO weights** — auto-downloaded by `ultralytics` on first run (`yolov8n.pt`, `yolov8n-pose.pt`).
   No action needed, just requires internet on first launch.

5. **Camera credentials** — create two files in the project root (not committed):
   ```
   echo -n "admin" > .camera_user
   echo -n "yourpassword" > .camera_pw
   chmod 600 .camera_user .camera_pw
   ```

6. **Network** — `CAMERA_IP`, `SOURCE_IP`, `PROXY_PORT` near the top of `app.py` assume this
   machine's specific network setup (a direct-cable interface bound to a fixed IP, proxied to
   avoid a Wi-Fi/cable subnet collision). Adjust these for the new machine's actual network.

7. **Run**
   ```
   python3 -u app.py
   ```
   Opens on `http://localhost:5055`.

## Data (not in this repo)

`events.db` (attendance/recognition database), `snapshots/` (face photos), and the camera
credential files hold real people's biometric data and are deliberately excluded via
`.gitignore`. Move these separately (e.g. `scp`/USB) if you want to carry history over instead
of starting fresh.

## Layout

Everything — proxy, database layer, camera stream + detection pipeline, Flask routes, and the
full frontend (HTML/CSS/JS) — lives in the single `app.py` file.
