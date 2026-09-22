# AGENTS.md

Shared context for any AI coding agent (Claude Code, Codex, Cursor, Copilot, etc.)
working in this repository. Read this before making changes.

## What this is

Offline human-detection + face-recognition system for a single Dahua IP camera
(DH-IPC-HDBW3241RP-ZAS). Pulls the raw RTSP feed directly and runs YOLOv8
(person detection) + YuNet/SFace (face detection + recognition) locally —
no vendor cloud/AI, no external API calls for inference.

A "new face" flow lets an operator assign a Name + Company to an unrecognized
face from the browser UI; recognition then persists across restarts via SQLite
(embeddings stored per person, not raw video).

## Layout — everything is in one file

`app.py` (~1345 lines) contains the *entire* application: RTSP proxy, SQLite
schema/helpers, model loading, the camera reader/inference threads, the Flask
routes, and the full frontend (HTML/CSS/JS as an inline Python string). There
is no build step and no separate frontend project. When editing the UI, you
are editing a Python multi-line string (`PAGE` / `DB_PAGE`) — watch for
`{`/`}` collisions if you ever convert it to an f-string, and keep JS/CSS
escaping in mind.

Rough section order in `app.py`:
1. RTSP proxy (`start_rtsp_proxy`) — binds outbound sockets to a specific NIC
   to dodge a subnet collision between Wi-Fi and the direct-cabled camera
   adapter. ffmpeg/ffprobe connect to `127.0.0.1:PROXY_PORT`, not the camera
   directly.
2. Config constants — thresholds, timeouts, paths. **Read the inline
   comments before changing any of these**; most encode a specific past bug
   and its root cause (see below).
3. SQLite schema + helpers (`init_db`, `db()`, person/company/embedding CRUD).
4. Model loading (YOLOv8, YOLOv8-pose, YuNet, SFace, MediaPipe Hands/FaceMesh).
5. `CameraStream` class — three daemon threads: reader (drains ffmpeg's raw
   frame pipe), inference (YOLO + face detect/recognize on the latest frame),
   and watchdog (force-reconnects on a stalled/unclosed connection). Video
   and detection overlays are served as two independent streams so smooth
   video never blocks on slow inference.
6. Flask routes — `/`, `/raw_feed` (MJPEG), `/detections` (JSON, polled by
   the browser to draw overlays on `<canvas>`), `/events`, `/pending*`
   (unknown-face queue: list/crop/assign/skip/acknowledge), `/database`
   (read-only DB explorer page), `/settings`, `/status`, `/rotate`,
   `/advanced`.

## Why the tuning constants look the way they do

Several constants (`MATCH_THRESHOLD`, `DEDUP_THRESHOLD`, `PENDING_ABSENCE_SEC`,
`RECOGNITION_GRACE_SEC`, `MIN_FACE_FRAC`, `MAX_EMBEDS_PER_PERSON`, etc.) were
moved from their "textbook" defaults after live debugging on this specific
camera (false face matches, duplicate sirens for one visit, faces judged too
small/far, gallery cap not actually enforced on load, etc.). The comment
directly above each constant in `app.py` explains the incident that drove the
current value — read it before retuning; don't revert to a "more standard"
value without understanding why it was moved away from.

## Networking / hardware assumptions (will need updating per machine)

- `CAMERA_IP`, `SOURCE_IP`, `PROXY_PORT` near the top of `app.py` assume a
  specific machine's network setup (a direct-cable NIC bound to a fixed IP,
  proxied to avoid a Wi-Fi/cable subnet collision on `192.168.1.x`).
- `FFMPEG_BIN` / `FFPROBE_BIN` are hardcoded absolute paths
  (`/opt/miniconda3/bin/...`) — update to match `which ffmpeg` on whatever
  machine runs this.
- Camera stream uses the **sub** stream (704x576) by default for detection
  speed, with the main stream (2304x1296) as fallback — see
  `RTSP_CANDIDATES`. Main-stream detection was measured at ~1.1–1.5 fps with
  no GPU; sub runs ~4x faster.
- `opencv-python`'s pip wheel has no FFmpeg support, so `cv2.VideoCapture`
  cannot open RTSP directly here — frames are piped in raw from the real
  system `ffmpeg` binary via `subprocess`, not decoded by OpenCV itself.

## Data & privacy — do not commit real data

`events.db`, `snapshots/`, `known_faces/`, `.camera_user`, `.camera_pw`, and
`*.onnx`/`*.pt` model weights are all gitignored deliberately. They contain
real people's biometric data (face embeddings, photos) and/or camera
credentials. Never add these to git, never print embeddings/credentials in
logs or commit messages, and don't loosen `.gitignore` for them without
explicit user instruction.

## Setup / running (see README.md for full detail)

- Python 3.11 via conda (`camera_ai_py311`); `pip install -r requirements.txt`.
- Needs system `ffmpeg`/`ffprobe` (not the opencv-python bundled build).
- `yunet.onnx` + `sface.onnx` (OpenCV Zoo) must be placed in the project root
  manually — not committed, not auto-downloaded.
- YOLO weights (`yolov8n.pt`, `yolov8n-pose.pt`) auto-download via
  `ultralytics` on first run.
- Camera credentials go in `.camera_user` / `.camera_pw` (root, chmod 600).
- Run: `python3 -u app.py` → `http://localhost:5055`.

## Schema that exists but isn't wired up yet

`products` and `stock_events` tables are created by `init_db()` for a planned
future stock/inventory pipeline (QR/barcode/OCR + object detection), and the
`/database` explorer page already renders them — but nothing currently writes
to them. Don't assume they're dead code if asked to build that feature; check
with the user before repurposing or dropping these tables.

## Conventions for agents editing this repo

- Keep the single-file structure unless the user explicitly asks for a
  refactor/split — this is a deliberate, small-project choice, not an
  oversight.
- Preserve the "why" comments on tuning constants; extend them (don't
  delete them) if you change a value, explaining what new observation
  justified the change.
- Threading model relies on `self.lock` (frame/state) and `gallery_lock`
  (recognition gallery) — hold the right lock when touching shared
  `CameraStream` state or `known_gallery`.
- No test suite exists in this repo currently; validate changes by running
  the app against the real camera (or reasoning carefully through the
  frame/threading logic) rather than assuming CI coverage.
