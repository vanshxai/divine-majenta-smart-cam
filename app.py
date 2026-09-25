"""
Offline human-detection + face-recognition framework for a single RTSP camera.
Pipeline: RTSP -> YOLO (person boxes) -> YuNet (face boxes) -> SFace (identity) -> SQLite -> browser.

New-face flow: an unrecognized face becomes a "pending" entry (deduped, so the same
person standing there doesn't spam it). The browser shows a popup with the face crop;
you type a Name + Company, hit Assign, and recognition updates immediately (no restart).
People + their face embeddings live in SQLite, so identity persists across restarts and
across moving the camera to a new location -- only the events' "location" tag changes.

Run:  python3 app.py
View: http://localhost:5055
"""
import os
import sys
import cv2
import json
import time
import uuid
import shutil
import socket
import sqlite3
import threading
import subprocess
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory

# ---- config ---------------------------------------------------------------
# When frozen by PyInstaller, data files (models) sit next to the executable /
# in the unpacked bundle dir; otherwise next to this source file. resource_path
# resolves either so the packaged Windows .exe finds yunet.onnx/sface.onnx/yolo.
if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)          # writable dir beside the .exe (db, snapshots, config)
    BUNDLE = getattr(sys, "_MEIPASS", BASE)          # read-only unpacked bundle (models baked in at build)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLE = BASE


def resource_path(name):
    """Model/data file: prefer one shipped next to the app (user-supplied), else the baked-in bundle copy."""
    p = os.path.join(BASE, name)
    return p if os.path.exists(p) else os.path.join(BUNDLE, name)


# config.json (optional, next to the app) overrides these defaults so the same build runs on any
# machine/network without editing source. See config.example.json. Camera credentials still live in
# the separate .camera_user/.camera_pw files (kept out of config so they're easy to gitignore).
_CFG = {}
_cfg_path = os.path.join(BASE, "config.json")
if os.path.exists(_cfg_path):
    try:
        with open(_cfg_path) as _f:
            _CFG = json.load(_f)
    except Exception as _e:
        print(f"[config] could not read config.json ({_e}); using defaults", flush=True)

CAMERA_IP = _CFG.get("camera_ip", "192.168.1.3")
HTTP_PORT = int(_CFG.get("http_port", 5055))

# The RTSP proxy is a Mac-specific workaround: this Mac's Wi-Fi (en0) and the direct-cabled camera
# adapter (en6) both use 192.168.1.x, so routing to CAMERA_IP is ambiguous and can time out. The proxy
# binds its outbound socket to en6's address to force the right path. On a normal single-network machine
# (e.g. the Windows box) this isn't needed -- set "use_proxy": false in config.json to connect directly.
USE_RTSP_PROXY = bool(_CFG.get("use_proxy", True))
SOURCE_IP = _CFG.get("source_ip", "192.168.1.50")  # en6 (direct-cable adapter); only used with the proxy
PROXY_PORT = int(_CFG.get("proxy_port", 15540))


def _relay(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass


def _handle_proxy_client(client_sock):
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        upstream.bind((SOURCE_IP, 0))
        upstream.connect((CAMERA_IP, 554))
    except OSError as e:
        print(f"[proxy] upstream connect via {SOURCE_IP} failed: {e}", flush=True)
        client_sock.close()
        return
    threading.Thread(target=_relay, args=(client_sock, upstream), daemon=True).start()
    threading.Thread(target=_relay, args=(upstream, client_sock), daemon=True).start()


def start_rtsp_proxy():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PROXY_PORT))
    srv.listen(8)
    print(f"[proxy] 127.0.0.1:{PROXY_PORT} -> {CAMERA_IP}:554 (via {SOURCE_IP})", flush=True)
    while True:
        client, _ = srv.accept()
        _handle_proxy_client(client)


if USE_RTSP_PROXY:
    threading.Thread(target=start_rtsp_proxy, daemon=True).start()
    time.sleep(0.3)  # let the proxy's listen socket come up before ffprobe/ffmpeg try it
    _RTSP_HOST = f"127.0.0.1:{PROXY_PORT}"
else:
    _RTSP_HOST = f"{CAMERA_IP}:554"  # direct connect -- normal single-network machines


def _read_secret(fname, default):
    p = os.path.join(BASE, fname)
    return open(p).read().strip() if os.path.exists(p) else default


CAMERA_USER = _read_secret(".camera_user", "admin")
CAMERA_PASS = _read_secret(".camera_pw", "admin")

# confirmed working via ffprobe on this camera (DH-IPC-HDBW3241RP-ZAS):
# main: 2304x1296@20fps, sub: 704x576@20fps -- sub used for speed (main ~1.1-1.5fps, sub ~4-5fps, no GPU).
# _RTSP_HOST is the proxy (127.0.0.1) or the camera directly, depending on config (see above). The Dahua
# path below is the common one; other vendors (Hikvision etc.) use different paths added at discovery time.
RTSP_CANDIDATES = [
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{_RTSP_HOST}/cam/realmonitor?channel=1&subtype=1",  # sub stream (preferred)
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{_RTSP_HOST}/cam/realmonitor?channel=1&subtype=0",  # main stream (fallback)
]
DB_PATH = os.path.join(BASE, "events.db")
SNAP_DIR = os.path.join(BASE, "snapshots")
KNOWN_DIR = os.path.join(BASE, "known_faces")  # legacy: subfolder per person, imported into the DB once
YUNET_MODEL = resource_path("yunet.onnx")
SFACE_MODEL = resource_path("sface.onnx")
MATCH_THRESHOLD = 0.38    # SFace's textbook default is 0.363. Was loosened to 0.30 to stop known faces
                          # flagging "unknown" -- but matching takes the BEST score across every stored
                          # reference photo per person, and Vansh alone has 28 of them (auto-enriched
                          # over time). More references = more chances for a stranger's face to score
                          # high against just ONE of them -- that's what caused an employee to match as
                          # Vansh. Raised past the textbook default to counter that "best of many" effect.
                          # The detector-confidence and face-size fixes already improve real match quality,
                          # so this shouldn't reintroduce false "unknown" the way the old 0.363 did.
DEDUP_THRESHOLD = 0.30    # looser than MATCH_THRESHOLD on purpose: "is this pending unknown probably the
                          # same still-unidentified person as that other pending entry". Was wrongly set
                          # to 0.45 (STRICTER than MATCH_THRESHOLD) -- that made it too easy to fragment
                          # one continuous unresolved visitor into several different "Unknown-N" cards,
                          # each with its own siren. A false merge here just means two different strangers
                          # briefly share one card, which is harmless; a false split re-triggers the alarm,
                          # which is the actual complaint -- so err toward merging.
RELOG_COOLDOWN_SEC = 60   # don't re-log the same known person more often than this
PENDING_ABSENCE_SEC = 25  # if an unassigned face hasn't been re-seen this long, treat them as having
                          # left the camera's view -- next time they appear it's a fresh alert (siren again).
                          # Was 10s, which was too short: with slower detection cadence (advanced modes,
                          # or just a face briefly turned away), a still-present person could go quiet for
                          # >10s, get treated as "gone", and re-trigger a fresh siren on the very next
                          # frame -- that's what was causing the same visit to alarm 2-3 times.
SKIP_COOLDOWN_SEC = 30    # after "Skip", don't re-prompt for the same face for this long
RECOGNITION_GRACE_SEC = 120  # an ALREADY-ACKNOWLEDGED unknown who briefly drops out of detection
                              # (bad angle, motion blur, walked just out of frame for a moment) gets
                              # silently revived under the same label within this window instead of
                              # spawning a brand-new alarm -- this is the actual fix for "siren firing
                              # multiple times for the same person". A real departure (longer than this)
                              # still re-alerts, as originally requested.
MAX_PENDING = 6
MIN_FACE_FRAC = 0.22      # a detected face's box height must be at least 22% of the frame height to
                          # count as "properly in view". One single frame at/above this size is enough
                          # -- no multi-frame wait. Below this, nothing happens: no label, no logging,
                          # no pending alert, no siren. Was 0.70 then 0.55, both calibrated off a
                          # deliberate close-up test -- measured live at NORMAL standing distance from
                          # this camera, a real face is only ~0.35 of frame height. 0.22 comfortably
                          # covers normal distance while still rejecting a tiny/distant/partial face.
FACE_CONFIRM_FRAMES = 1   # how many consecutive frames at MIN_FACE_FRAC+ are required before judging
                          # known/unknown -- 1 means decide immediately, no waiting.
FACE_TRACK_TIMEOUT_SEC = 2.0  # a face-position track not re-seen this long is dropped; reappearing
                               # later starts the confirmation count over from zero.
STALL_TIMEOUT_SEC = 10    # no frame at all for this long -> assume the connection is stalled (cable
                          # pulled, camera powered off) and force a reconnect. See _watchdog_loop.
CAMERA_LABEL = "Camera 1"  # single camera for now; multi-camera support will parameterize this per instance
DEFAULT_COMPANY = "Divine Technologies"
ROTATE_180 = False  # checked directly against a raw frame -- the camera is NOT physically upside
                    # down (an earlier guess from a small screenshot was wrong); leave unrotated.
MAX_EMBEDDINGS_PER_PENDING = 5  # accumulate a few samples of the same pending face (different head
                                 # angles/expressions) instead of just one -- much more robust matching
MAX_EMBEDS_PER_PERSON = 5         # cap on stored reference samples per known person. Was 20 (then found
                                  # to actually be 28, uncapped -- see load_known_gallery fix). More
                                  # samples = more surface area for a stranger to false-match against
                                  # just one of them ("best score across all references" effect) -- 5
                                  # keeps the gallery tight instead of overfitting to accumulated noise.
AUTO_ENRICH_COOLDOWN_SEC = 300   # while confidently recognized, bank one more sample at most this often --
                                  # over time this naturally collects both B&W/IR and color samples of the
                                  # same person without needing you to manually re-add them
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;3000000")

# ---- storage ----------------------------------------------------------------
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)


def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    con = db()
    con.execute("""CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        company_id INTEGER REFERENCES companies(id),
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL REFERENCES people(id),
        embedding BLOB NOT NULL,
        photo_path TEXT,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        snapshot TEXT,
        person_id INTEGER,
        location TEXT
    )""")
    # migrate an older events table (pre-person_id/location) if it exists from a prior run
    cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    for col, decl in (("person_id", "INTEGER"), ("location", "TEXT")):
        if col not in cols:
            con.execute(f"ALTER TABLE events ADD COLUMN {col} {decl}")
    # schema for the future stock/inventory pipeline (QR/barcode/OCR + object detection) -- tables only,
    # nothing writes to these yet. Ready so that pipeline can be built later without another migration.
    con.execute("""CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS stock_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        product_id INTEGER REFERENCES products(id),
        action TEXT,          -- 'taken' | 'added' | 'scanned', etc.
        quantity INTEGER,
        person_id INTEGER REFERENCES people(id),
        snapshot TEXT,
        location TEXT
    )""")
    # attendance: one row per person per day. check_in is set once (first sighting in the morning
    # window); check_out tracks the latest sighting in the evening window. UNIQUE(person_id, date)
    # is what enforces "log each person only once per day" no matter how often they're seen.
    con.execute("""CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER REFERENCES people(id),
        name TEXT,
        date TEXT NOT NULL,           -- YYYY-MM-DD (local)
        check_in TEXT,                -- ISO time, first seen in the morning window
        check_out TEXT,               -- ISO time, last seen in the evening window
        check_in_snapshot TEXT,
        check_out_snapshot TEXT,
        UNIQUE(person_id, date)
    )""")
    con.commit()
    if get_setting(con, "default_company") is None:
        set_setting(con, "default_company", DEFAULT_COMPANY)
    if get_setting(con, "location") is None:
        set_setting(con, "location", "")
    # attendance window defaults (editable in the UI). Times are local "HH:MM", 24-hour.
    for key, default in (("att_enabled", "1"),
                          ("att_morning_start", "08:45"), ("att_morning_end", "09:15"),
                          ("att_evening_start", "16:45"), ("att_evening_end", "17:15")):
        if get_setting(con, key) is None:
            set_setting(con, key, default)
    con.close()


def get_setting(con, key, default=None):
    row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(con, key, value):
    con.execute("INSERT INTO settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))
    con.commit()


def get_or_create_company(con, name):
    name = (name or "").strip()
    if not name:
        return None
    row = con.execute("SELECT id FROM companies WHERE name=?", (name,)).fetchone()
    if row:
        return row[0]
    cur = con.execute("INSERT INTO companies (name, created_at) VALUES (?,?)",
                       (name, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    return cur.lastrowid


def create_person(con, name, company_id):
    cur = con.execute("INSERT INTO people (name, company_id, created_at) VALUES (?,?,?)",
                       (name, company_id, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    return cur.lastrowid


def add_embedding(con, person_id, embedding, photo_path=None):
    con.execute("INSERT INTO face_embeddings (person_id, embedding, photo_path, created_at) VALUES (?,?,?,?)",
                (person_id, np.asarray(embedding, dtype=np.float32).tobytes(), photo_path,
                 datetime.now().isoformat(timespec="seconds")))
    # keep only the MAX_EMBEDS_PER_PERSON most recent rows for this person -- enforced in the DB itself
    # (not just the in-memory gallery) so the two can never drift apart again like Vansh's 28-vs-20 did.
    con.execute("""DELETE FROM face_embeddings WHERE person_id=? AND id NOT IN (
                       SELECT id FROM face_embeddings WHERE person_id=? ORDER BY id DESC LIMIT ?)""",
                (person_id, person_id, MAX_EMBEDS_PER_PERSON))
    con.commit()


def log_event(name, kind, snapshot_path, person_id=None, location=None):
    con = db()
    con.execute("INSERT INTO events (ts, name, kind, snapshot, person_id, location) VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), name, kind, snapshot_path, person_id, location))
    con.commit()
    con.close()


def recent_events(limit=2000):
    con = db()
    rows = con.execute("""SELECT e.ts, e.name, e.kind, e.snapshot, e.location, c.name
                           FROM events e
                           LEFT JOIN people p ON p.id = e.person_id
                           LEFT JOIN companies c ON c.id = p.company_id
                           ORDER BY e.id DESC LIMIT ?""", (limit,)).fetchall()
    con.close()
    return [{"ts": r[0], "name": r[1], "kind": r[2], "snapshot": r[3], "location": r[4], "company": r[5]}
            for r in rows]


# ---- attendance -------------------------------------------------------------
def get_attendance_config(con=None):
    own = con is None
    if own:
        con = db()
    cfg = {
        "enabled": get_setting(con, "att_enabled", "1") == "1",
        "morning_start": get_setting(con, "att_morning_start", "08:45"),
        "morning_end": get_setting(con, "att_morning_end", "09:15"),
        "evening_start": get_setting(con, "att_evening_start", "16:45"),
        "evening_end": get_setting(con, "att_evening_end", "17:15"),
    }
    if own:
        con.close()
    return cfg


def current_attendance_phase(cfg=None, now=None):
    """Returns 'morning', 'evening', or None depending on whether the current local time falls
    inside a configured window. Comparison is on "HH:MM" strings, which sort correctly for a
    same-day 24h clock."""
    if cfg is None:
        cfg = get_attendance_config()
    if not cfg["enabled"]:
        return None
    hm = (now or datetime.now()).strftime("%H:%M")
    if cfg["morning_start"] <= hm <= cfg["morning_end"]:
        return "morning"
    if cfg["evening_start"] <= hm <= cfg["evening_end"]:
        return "evening"
    return None


def mark_attendance(person_id, name, snapshot, phase, now):
    """Record attendance for a recognized known person, gated to the active window.
    Morning -> set check_in once (first sighting). Evening -> keep check_out at the latest sighting.
    UNIQUE(person_id, date) means a person is only ever one row per day no matter how often seen."""
    if person_id is None or phase is None:
        return
    date = now.strftime("%Y-%m-%d")
    ts = now.isoformat(timespec="seconds")
    con = db()
    con.execute("""INSERT OR IGNORE INTO attendance (person_id, name, date) VALUES (?,?,?)""",
                (person_id, name, date))
    if phase == "morning":
        # only fills check_in if it's still empty -- so the FIRST morning sighting wins, later ones ignored
        con.execute("""UPDATE attendance SET check_in=?, check_in_snapshot=?, name=?
                       WHERE person_id=? AND date=? AND check_in IS NULL""",
                    (ts, snapshot, name, person_id, date))
    else:  # evening -> always advance check_out to the most recent sighting (when they left)
        con.execute("""UPDATE attendance SET check_out=?, check_out_snapshot=?, name=?
                       WHERE person_id=? AND date=?""",
                    (ts, snapshot, name, person_id, date))
    con.commit()
    con.close()


def attendance_for_date(date=None):
    date = date or datetime.now().strftime("%Y-%m-%d")
    con = db()
    rows = con.execute("""SELECT a.name, c.name, a.check_in, a.check_out,
                                 a.check_in_snapshot, a.check_out_snapshot
                          FROM attendance a
                          LEFT JOIN people p ON p.id = a.person_id
                          LEFT JOIN companies c ON c.id = p.company_id
                          WHERE a.date=? ORDER BY a.check_in IS NULL, a.check_in""", (date,)).fetchall()
    con.close()
    return [{"name": r[0], "company": r[1], "check_in": r[2], "check_out": r[3],
             "check_in_snapshot": r[4], "check_out_snapshot": r[5]} for r in rows]


# ---- models -----------------------------------------------------------------
print("[init] loading YOLO...", flush=True)
from ultralytics import YOLO
yolo = YOLO(resource_path("yolov8n.pt"))  # official Ultralytics weights (shipped next to app or in bundle)
PERSON_CLASS_ID = [k for k, v in yolo.names.items() if v == "person"][0]

print("[init] loading YuNet + SFace...", flush=True)
face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.5)
# was 0.7 -- live testing showed real-face confidence bouncing between 0.3 and 0.87 frame to frame
# (motion, angle, distance) on this camera; 0.7 was rejecting a face outright before our own
# MIN_FACE_FRAC size gate even got a chance to see it. 0.5 lets more real candidates through; the
# size gate + SFace match/no-match downstream still filter out anything that isn't a real match.
face_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")


def embed_face(bgr_frame, face_box_5pt):
    aligned = face_recognizer.alignCrop(bgr_frame, face_box_5pt)
    return face_recognizer.feature(aligned)


def cosine(a, b):
    return face_recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)


def iou(box_a, box_b):
    """Standard box overlap ratio, used only for short-term face-position tracking (not identity)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# known_gallery: {person_id: {"name": str, "company": str|None, "embeds": [np.ndarray,...]}}
known_gallery = {}
gallery_lock = threading.Lock()  # guards known_gallery + last_seen_at against the assign-vs-recognize race
last_seen_at = {}  # person_id (or "Unknown-N") -> timestamp, for the relog cooldown
last_enriched_at = {}  # person_id -> timestamp, throttles auto-enrichment (see AUTO_ENRICH_COOLDOWN_SEC)


def enrich_person(person_id, embedding, now):
    """Bank an extra reference sample for an already-known, confidently-matched person.
    This is what makes recognition keep working across lighting changes (B&W/IR at night vs
    color in daylight) without you having to manually re-enroll -- samples accumulate from
    whatever conditions the person is actually seen under, over time."""
    if now - last_enriched_at.get(person_id, 0) < AUTO_ENRICH_COOLDOWN_SEC:
        return
    last_enriched_at[person_id] = now
    con = db()
    add_embedding(con, person_id, embedding)
    con.close()
    with gallery_lock:
        entry = known_gallery.get(person_id)
        if entry:
            entry["embeds"].append(embedding)
            if len(entry["embeds"]) > MAX_EMBEDS_PER_PERSON:
                entry["embeds"] = entry["embeds"][-MAX_EMBEDS_PER_PERSON:]


def _import_legacy_folder_once(con):
    """One-time import of known_faces/<name>/*.jpg into the DB, if the DB has no people yet."""
    if con.execute("SELECT COUNT(*) FROM people").fetchone()[0] > 0:
        return
    if not os.path.isdir(KNOWN_DIR):
        return
    for person in sorted(os.listdir(KNOWN_DIR)):
        pdir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(pdir):
            continue
        photos = [f for f in os.listdir(pdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not photos:
            continue
        person_id = create_person(con, person, None)
        count = 0
        for fn in photos:
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                continue
            face_detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = face_detector.detect(img)
            if faces is None or len(faces) == 0:
                continue
            add_embedding(con, person_id, embed_face(img, faces[0]), os.path.join(pdir, fn))
            count += 1
        print(f"[gallery] imported legacy folder '{person}': {count} photo(s)", flush=True)


def load_known_gallery():
    con = db()
    _import_legacy_folder_once(con)
    gallery = {}
    # most recent MAX_EMBEDS_PER_PERSON first per person -- this cap was already meant to apply (it's
    # enforced when NEW embeddings get added at runtime) but was never applied on startup load, so a
    # restart silently brought back every embedding ever stored (28 for Vansh, not the intended 20).
    # More stored references = more chances for a stranger to false-match against just one of them.
    rows = con.execute("""SELECT p.id, p.name, c.name, f.embedding FROM people p
                           LEFT JOIN companies c ON c.id = p.company_id
                           JOIN face_embeddings f ON f.person_id = p.id
                           ORDER BY f.id DESC""").fetchall()
    con.close()
    for person_id, name, company, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
        entry = gallery.setdefault(person_id, {"name": name, "company": company, "embeds": []})
        if len(entry["embeds"]) < MAX_EMBEDS_PER_PERSON:
            entry["embeds"].append(emb)
    for pid, e in gallery.items():
        print(f"[gallery] {e['name']} ({e['company'] or 'no company'}): {len(e['embeds'])} reference photo(s)", flush=True)
    if not gallery:
        print("[gallery] empty — no one enrolled yet; unknown faces will show a name-assignment popup", flush=True)
    return gallery


def match_identity(embedding):
    """Returns (person_id, name, company, score). person_id is None for no match."""
    with gallery_lock:
        best_pid, best_name, best_company, best_score = None, "Unknown", None, -1.0
        for pid, e in known_gallery.items():
            for ref in e["embeds"]:
                score = cosine(embedding, ref)
                if score > best_score:
                    best_pid, best_name, best_company, best_score = pid, e["name"], e["company"], score
    if best_score >= MATCH_THRESHOLD:
        return best_pid, best_name, best_company, best_score
    return None, "Unknown", None, best_score


# ---- camera reader thread ---------------------------------------------------
# This opencv-python wheel was built with no FFmpeg support (avcodec/avformat: NO),
# so cv2.VideoCapture can never open RTSP here. Pipe raw frames from the real
# system ffmpeg binary instead -- same transport that ffprobe already proved works.
# Resolution order: config.json override -> a copy shipped next to the app (Windows: ffmpeg.exe in
# an "ffmpeg" subfolder) -> the system one on PATH. Keeps the Mac dev path and the Windows exe both working.
def _resolve_bin(name, cfg_key):
    override = _CFG.get(cfg_key)
    if override and os.path.exists(override):
        return override
    exe = name + (".exe" if os.name == "nt" else "")
    for cand in (os.path.join(BASE, exe), os.path.join(BASE, "ffmpeg", exe),
                 "/opt/miniconda3/bin/" + name):
        if os.path.exists(cand):
            return cand
    return shutil.which(name) or exe  # fall back to PATH; bare name lets ffmpeg errors surface clearly


FFMPEG_BIN = _resolve_bin("ffmpeg", "ffmpeg_path")
FFPROBE_BIN = _resolve_bin("ffprobe", "ffprobe_path")
print(f"[init] ffmpeg: {FFMPEG_BIN}", flush=True)


def probe_size(url):
    out = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-rtsp_transport", "tcp",
         "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", url],
        capture_output=True, text=True, timeout=6,
    )
    line = out.stdout.strip()
    if not line or "," not in line:
        return None
    w, h = line.split(",")[:2]
    return int(w), int(h)


class CameraStream:
    def __init__(self):
        self.proc = None
        self.w = self.h = None
        self.url_used = None
        self.frame = None
        self.detections = []
        self.det_frame_size = (None, None)
        self.det_ts = 0
        self.lock = threading.Lock()
        self.running = True
        self.status = "connecting"
        self.rotate_180 = ROTATE_180  # runtime-toggleable via the UI button / POST /rotate
        self.pending = {}         # pending_id -> {embeddings, crop, first_seen, last_seen, label, acknowledged}
        self.skip_cooldown = []   # [(embedding, ts), ...] recently-dismissed faces, don't re-prompt yet
        self.recently_dismissed = []  # acknowledged pending entries that expired -- kept briefly so a
                                       # short gap in detection doesn't re-alarm (see RECOGNITION_GRACE_SEC)
        self.next_unknown_id = 1  # -> "Unknown-1", "Unknown-2", ... stable per still-present face
        self.face_tracks = []     # [{box, streak, last_seen}] -- short-term position tracking used only
                                   # to require a face be properly in view for a few frames running before
                                   # we judge known/unknown at all (see FACE_CONFIRM_FRAMES)
        self.last_frame_at = time.time()  # watchdog uses this to notice a stalled (not cleanly closed)
                                           # connection -- see _watchdog_loop
        threading.Thread(target=self._connect_and_loop, daemon=True).start()
        threading.Thread(target=self._inference_loop, daemon=True).start()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _watchdog_loop(self):
        """A yanked cable / powered-off camera often doesn't close the TCP connection cleanly --
        no FIN/RST arrives, so ffmpeg's stdout.read() in the reader thread just blocks forever and
        the existing 'read failed, reconnecting' path never runs. This is what was actually stopping
        auto-reconnect from noticing a real disconnect. Here we watch wall-clock time since the last
        successfully read frame; once it's been too long, force-kill the stuck ffmpeg process -- that
        closes its stdout pipe, which unblocks the reader thread's read() with a short read, which
        then falls into the normal reconnect path on its own."""
        while self.running:
            time.sleep(3)
            if self.proc is not None and time.time() - self.last_frame_at > STALL_TIMEOUT_SEC:
                print(f"[camera] no frame for {STALL_TIMEOUT_SEC}s -- connection stalled "
                      f"(likely unplugged/powered off), forcing reconnect", flush=True)
                self.status = "stalled -- reconnecting..."
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def _try_open(self):
        for url in RTSP_CANDIDATES:
            print(f"[camera] probing {url}", flush=True)
            try:
                size = probe_size(url)
            except Exception as e:
                print(f"[camera]   -> probe error {e}", flush=True)
                continue
            if not size:
                print("[camera]   -> failed", flush=True)
                continue
            w, h = size
            print(f"[camera]   -> OK {w}x{h}", flush=True)
            proc = subprocess.Popen(
                [FFMPEG_BIN, "-rtsp_transport", "tcp", "-i", url,
                 "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=w * h * 3 * 2,
            )
            return proc, url, w, h
        return None, None, None, None

    def _connect_and_loop(self):
        """Reader thread: only job is to drain ffmpeg's pipe as fast as possible and
        keep the newest raw frame. Never blocks on inference -- that's a separate thread."""
        frame_bytes = None
        while self.running:
            if self.proc is None:
                self.proc, self.url_used, self.w, self.h = self._try_open()
                if self.proc is None:
                    self.status = "no candidate RTSP path worked; edit RTSP_CANDIDATES in app.py"
                    time.sleep(5)
                    continue
                self.status = f"connected: {self.url_used} ({self.w}x{self.h})"
                frame_bytes = self.w * self.h * 3
                self.last_frame_at = time.time()  # fresh baseline -- give the first frame a moment to arrive
            raw = self.proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                print("[camera] read failed, reconnecting...", flush=True)
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None
                time.sleep(2)
                continue
            self.last_frame_at = time.time()
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 3)
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            with self.lock:
                self.frame = frame

    def _inference_loop(self):
        """Separate thread: always grabs whatever the LATEST frame is and runs
        YOLO/face detection on it, dropping any frames produced while it was busy."""
        last_processed_id = None
        while self.running:
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            t0 = time.time()
            self._process(frame)
            dt = time.time() - t0
            if int(dt * 10) != last_processed_id:
                print(f"[infer] {dt*1000:.0f}ms/frame (~{1/dt:.1f} fps ceiling)", flush=True)
                last_processed_id = int(dt * 10)

    # ---- face-in-view confirmation (runs before any known/unknown judgment) --
    def _confirm_face_track(self, box, now):
        """Returns the current consecutive-frame streak for whichever track this box belongs to
        (matched by position overlap, not identity -- identity isn't decided yet at this point).
        Only called for boxes already passing MIN_FACE_FRAC, so a streak only grows while the face
        stays properly sized in frame."""
        self.face_tracks = [t for t in self.face_tracks if now - t["last_seen"] < FACE_TRACK_TIMEOUT_SEC]
        best, best_iou = None, 0.3  # 0.3 IoU minimum to count as "the same face" between frames
        for t in self.face_tracks:
            score = iou(box, t["box"])
            if score > best_iou:
                best, best_iou = t, score
        if best:
            best["box"], best["last_seen"] = box, now
            best["streak"] += 1
            return best["streak"]
        self.face_tracks.append({"box": box, "last_seen": now, "streak": 1})
        return 1

    # ---- pending (new-face) queue -------------------------------------------
    def _find_pending_match(self, embedding):
        for pid, p in self.pending.items():
            if max(cosine(embedding, e) for e in p["embeddings"]) >= DEDUP_THRESHOLD:
                return pid
        return None

    def _in_skip_cooldown(self, embedding, now):
        self.skip_cooldown = [(e, t) for e, t in self.skip_cooldown if now - t < SKIP_COOLDOWN_SEC]
        return any(cosine(embedding, e) >= DEDUP_THRESHOLD for e, _ in self.skip_cooldown)

    def _expire_pending(self, now):
        """Called every inference cycle regardless of what's in the current frame -- removes any
        pending face not re-seen in a while (they've walked out of view). An UNACKNOWLEDGED entry is
        just dropped -- nobody dealt with it, so a reappearance should alert again right away. An
        ACKNOWLEDGED entry is instead parked in recently_dismissed for RECOGNITION_GRACE_SEC: if it's
        really still the same person (brief detection gap, not a real departure), _offer_pending revives
        it silently below. Only a gap longer than the grace period counts as a real departure -> fresh,
        unacknowledged alert -> siren again, as originally requested."""
        with self.lock:
            for pid in [k for k, p in self.pending.items() if now - p["last_seen"] > PENDING_ABSENCE_SEC]:
                p = self.pending.pop(pid)
                if p["acknowledged"]:
                    self.recently_dismissed.append({"embeddings": p["embeddings"], "label": p["label"],
                                                      "expire_at": now + RECOGNITION_GRACE_SEC})
            self.recently_dismissed = [d for d in self.recently_dismissed if d["expire_at"] > now]

    @staticmethod
    def _make_crop(frame, x, y, fw, fh):
        pad = int(0.3 * max(fw, fh))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        crop = frame[y0:y + fh + pad, x0:x + fw + pad]
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) if crop.size > 0 else (False, None)
        return buf.tobytes() if ok else None

    def _find_recently_dismissed_match(self, embedding):
        for d in self.recently_dismissed:
            if max(cosine(embedding, e) for e in d["embeddings"]) >= DEDUP_THRESHOLD:
                return d
        return None

    def _offer_pending(self, frame, x, y, fw, fh, embedding, now):
        """Returns the stable label ("Unknown-N") for whichever pending cluster this face belongs
        to, creating one if needed. Returns None if this sighting was suppressed (skip-cooldown /
        too many pending already) -- caller falls back to a generic label in that case."""
        with self.lock:
            existing = self._find_pending_match(embedding)
            if existing:
                p = self.pending[existing]
                p["last_seen"] = now
                if len(p["embeddings"]) < MAX_EMBEDDINGS_PER_PENDING:
                    p["embeddings"].append(embedding)  # extra sample of this same face -> sturdier match later
                return p["label"]
            if self._in_skip_cooldown(embedding, now) or len(self.pending) >= MAX_PENDING:
                return None
            # not currently pending -- but if this is really the same person as one that was already
            # acknowledged and briefly dropped out of detection, revive it silently (stays acknowledged,
            # no new siren) instead of raising a fresh alert.
            revived = self._find_recently_dismissed_match(embedding)
            if revived:
                self.recently_dismissed.remove(revived)
                pid = uuid.uuid4().hex[:12]
                embeddings = (revived["embeddings"] + [embedding])[-MAX_EMBEDDINGS_PER_PENDING:]
                self.pending[pid] = {"embeddings": embeddings, "crop": self._make_crop(frame, x, y, fw, fh),
                                      "first_seen": now, "last_seen": now, "label": revived["label"],
                                      "acknowledged": True}
                return revived["label"]
            pid = uuid.uuid4().hex[:12]
            label = f"Unknown-{self.next_unknown_id}"
            self.next_unknown_id += 1
            self.pending[pid] = {"embeddings": [embedding], "crop": self._make_crop(frame, x, y, fw, fh),
                                  "first_seen": now, "last_seen": now, "label": label, "acknowledged": False}
            return label

    def get_pending_list(self):
        with self.lock:
            return sorted(
                [{"id": pid, "label": p["label"], "camera": CAMERA_LABEL, "first_seen": p["first_seen"],
                  "acknowledged": p["acknowledged"], "has_crop": p["crop"] is not None}
                 for pid, p in self.pending.items()],
                key=lambda x: -x["first_seen"])

    def acknowledge_pending(self, pid):
        with self.lock:
            p = self.pending.get(pid)
            if p:
                p["acknowledged"] = True
        return p is not None

    def get_pending_crop(self, pid):
        with self.lock:
            p = self.pending.get(pid)
            return p["crop"] if p else None

    def assign_pending(self, pid, name, company):
        """If `name` matches an existing person (e.g. re-adding yourself after being missed
        under different lighting -- B&W/IR vs color), MERGE these embeddings into that same
        person instead of creating a duplicate. That's what actually fixes cross-lighting
        recognition: the person ends up with reference samples from both conditions."""
        with self.lock:
            p = self.pending.pop(pid, None)
        if p is None:
            return None
        con = db()
        existing = con.execute("SELECT id, company_id FROM people WHERE lower(name)=lower(?)", (name,)).fetchone()
        if existing:
            person_id, existing_company_id = existing
            company_id = get_or_create_company(con, company) if company else existing_company_id
            if company_id != existing_company_id:
                con.execute("UPDATE people SET company_id=? WHERE id=?", (company_id, person_id))
                con.commit()
        else:
            company_id = get_or_create_company(con, company)
            person_id = create_person(con, name, company_id)
        for emb in p["embeddings"]:
            add_embedding(con, person_id, emb)
        if company:
            set_setting(con, "default_company", company)
        company_row = con.execute("SELECT name FROM companies WHERE id=?", (company_id,)).fetchone()
        company_name = company_row[0] if company_row else None
        con.close()
        with gallery_lock:
            entry = known_gallery.setdefault(person_id, {"name": name, "company": company_name, "embeds": []})
            entry["company"] = company_name
            entry["embeds"].extend(p["embeddings"])
            if len(entry["embeds"]) > MAX_EMBEDS_PER_PERSON:
                entry["embeds"] = entry["embeds"][-MAX_EMBEDS_PER_PERSON:]  # keep most recent samples
        return {"person_id": person_id, "name": name, "company": company_name,
                "samples": len(p["embeddings"]), "merged": bool(existing)}

    def skip_pending(self, pid):
        with self.lock:
            p = self.pending.pop(pid, None)
            if p:
                self.skip_cooldown.append((p["embeddings"][-1], time.time()))
        return p is not None

    # ---- inference -----------------------------------------------------------
    def _process(self, frame):
        """Computes detections as plain coordinates -- does NOT draw on the frame.
        Video (self.frame, raw) and detections (self.detections, boxes+labels) are
        served as two separate streams; the browser overlays them, so video stays
        smooth no matter how slow this is."""
        h, w = frame.shape[:2]
        dets = []
        con = db()
        location = get_setting(con, "location", "")
        con.close()

        # 1. person detection
        results = yolo.predict(frame, classes=[PERSON_CLASS_ID], verbose=False, conf=0.4)
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            dets.append({"type": "person", "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], "label": "person"})

        # 2. face detection + recognition (independent pass, whole frame)
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(frame)
        now = time.time()
        self._expire_pending(now)  # runs every cycle, even with zero faces this frame -- so someone
                                    # who has walked out of view gets marked "gone" promptly

        # RECORDING GATE: recognition + live overlay run all day, but NOTHING is written to disk/DB
        # outside a configured attendance window -- no event rows, no snapshot files, no enrichment.
        # rec_phase is "morning"/"evening" when a window is open, else None. Computed once per frame.
        rec_phase = current_attendance_phase()

        for f in (faces if faces is not None else []):
            x, y, fw, fh = f[:4].astype(int)

            if fh < MIN_FACE_FRAC * h:
                continue  # too small / just stepping into frame -- not properly in view yet, ignore entirely
            streak = self._confirm_face_track((x, y, fw, fh), now)
            if streak < FACE_CONFIRM_FRAMES:
                continue  # seen, but not for long enough yet to trust a judgment -- wait, do nothing

            embedding = embed_face(frame, f)
            person_id, name, company, score = match_identity(embedding)
            kind = "known" if person_id is not None else "unknown"

            if kind == "unknown":
                label = self._offer_pending(frame, x, y, fw, fh, embedding, now) or "Unknown"
                display_label, log_name = label, label
            else:
                display_label = f"{name} — {company}" if company else name
                log_name = name
                if rec_phase:
                    enrich_person(person_id, embedding, now)  # bank a sample -> hardens against lighting
                                                               # changes; only while recording is active

            dets.append({"type": "face", "kind": kind, "box": [int(x), int(y), int(fw), int(fh)],
                         "label": display_label})

            # everything below writes to disk/DB -- skip entirely unless an attendance window is open
            if not rec_phase:
                continue

            cooldown_key = person_id if person_id is not None else log_name
            if now - last_seen_at.get(cooldown_key, 0) > RELOG_COOLDOWN_SEC:
                last_seen_at[cooldown_key] = now
                ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"{log_name}_{ts_tag}.jpg"
                crop = frame[max(0, y):y + fh, max(0, x):x + fw]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(SNAP_DIR, snap_name), crop)
                log_event(log_name, kind, snap_name, person_id=person_id, location=location)

                # attendance: known people only. check_in is set-once (morning), check_out tracks the
                # latest evening sighting -- so a person seen many times is recorded once per window.
                if person_id is not None:
                    mark_attendance(person_id, name, snap_name, rec_phase, datetime.now())

        with self.lock:
            self.detections = dets
            self.det_frame_size = (w, h)
            self.det_ts = now

    def get_raw_jpeg(self):
        """Latest camera frame, undecorated -- this is what keeps the video smooth,
        since it never waits on YOLO/face inference."""
        with self.lock:
            frame = self.frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def get_detections(self):
        with self.lock:
            w, h = self.det_frame_size
            return {"detections": self.detections, "frame_w": w, "frame_h": h, "ts": self.det_ts}


init_db()
known_gallery = load_known_gallery()
camera = CameraStream()

# ---- web app ------------------------------------------------------------
app = Flask(__name__)

PAGE = """<!doctype html><html><head><title>camera_ai</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:0;display:flex}
#log{width:360px;padding:12px;height:100vh;box-sizing:border-box;display:flex;flex-direction:column;overflow:hidden}
#log h3{margin:0 0 8px 0;flex-shrink:0} .row{border-bottom:1px solid #333;padding:6px 0;font-size:13px}
.known{color:#4caf50} .unknown{color:#e53935}
.locSection{flex-shrink:0;margin-bottom:16px}
#pendingSection{flex-shrink:0;max-height:220px;display:flex;flex-direction:column;margin-bottom:16px}
#eventsSection{flex:1;min-height:0;display:flex;flex-direction:column}
.scrollbox{overflow-y:auto;min-height:0;scrollbar-width:thin;scrollbar-color:#555 #1a1a1a}
.scrollbox::-webkit-scrollbar{width:8px}
.scrollbox::-webkit-scrollbar-track{background:#1a1a1a}
.scrollbox::-webkit-scrollbar-thumb{background:#555;border-radius:4px}
#pendingList.scrollbox{flex:1} #rows.scrollbox{flex:1}
#stage{position:relative;width:calc(100vw - 360px);height:100vh;background:#000}
#stage img,#stage canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
#stage canvas{pointer-events:none}
#fps{position:absolute;top:6px;left:6px;font-size:12px;color:#0f0;background:#0008;padding:2px 6px;border-radius:3px}
#rotateBtn{position:absolute;top:6px;right:6px;z-index:2}
#attBtn{position:absolute;top:6px;right:150px;z-index:2}
#dbBtn{position:absolute;top:6px;right:300px;z-index:2;text-decoration:none;display:inline-block}
#attPill{position:absolute;top:8px;left:70px;z-index:2;font-size:13px;font-weight:bold;
         padding:4px 10px;border-radius:12px;background:#333;color:#aaa}
#attPill.open{background:#2e7d32;color:#fff}
#locbar{display:flex;gap:6px}
#locbar input{flex:1;background:#222;border:1px solid #444;color:#eee;padding:5px;border-radius:4px}
button{background:#333;border:1px solid #555;color:#eee;padding:5px 10px;border-radius:4px;cursor:pointer}
button:hover{background:#444}
#pendingList{display:flex;flex-direction:column;gap:8px}
.pcard{display:flex;align-items:center;gap:8px;background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:6px;flex-shrink:0}
.pcard.alerting{border-color:#e53935;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{background:#1a1a1a}50%{background:#3a1414}}
.pcard img{width:40px;height:40px;object-fit:cover;border-radius:4px}
.pcard span{flex:1;font-size:12px;color:#aaa}
#modal{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:10}
#card{background:#1a1a1a;border:1px solid #444;border-radius:10px;padding:20px;width:300px;text-align:center}
#card img{width:180px;height:180px;object-fit:cover;border-radius:8px;border:2px solid #555}
#card input{width:100%;box-sizing:border-box;margin-top:10px;padding:8px;background:#222;border:1px solid #444;color:#eee;border-radius:4px}
#card .btns{display:flex;gap:8px;margin-top:12px}
#card .btns button{flex:1}
#card .primary{background:#2e7d32;border-color:#2e7d32}
#attModal{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:10}
#attCard{background:#1a1a1a;border:1px solid #444;border-radius:10px;padding:22px;width:380px}
#attCard .attRow{display:flex;align-items:center;gap:8px;font-size:14px;margin-bottom:12px}
#attCard .attGrid{display:grid;grid-template-columns:1fr auto 1fr auto;gap:8px;align-items:center;font-size:13px;color:#bbb}
#attCard input[type=time]{background:#222;border:1px solid #444;color:#eee;padding:5px;border-radius:4px}
#attCard .btns{display:flex;gap:8px;margin-top:14px}
#attCard .btns button{flex:1}
#attCard .primary{background:#2e7d32;border-color:#2e7d32}
.attItem{display:flex;justify-content:space-between;font-size:13px;padding:6px 4px;border-bottom:1px solid #222}
.attItem .nm{font-weight:bold;color:#eee}
.attItem .tm{color:#9c9;font-variant-numeric:tabular-nums}
.attItem .out{color:#c99}
</style></head>
<body>
<div id="stage">
  <img id="vid" src="/raw_feed">
  <canvas id="overlay"></canvas>
  <div id="fps"></div>
  <div id="attPill">Attendance: —</div>
  <a id="dbBtn" href="/database" target="_blank"><button>🗄 Database</button></a>
  <button id="rotateBtn" onclick="toggleRotate()">⟳ Rotate 180°</button>
  <button id="attBtn" onclick="openAttendance()">🕐 Attendance</button>
</div>
<div id="log">
  <div class="locSection">
    <h3>Camera location</h3>
    <div id="locbar"><input id="locInput" placeholder="e.g. Front Office, Building A"><button onclick="saveLocation()">Save</button></div>
  </div>
  <div id="pendingSection">
    <h3>Unknown faces</h3>
    <div id="pendingList" class="scrollbox"><span style="font-size:12px;color:#666">none right now</span></div>
  </div>
  <div id="eventsSection">
    <h3>Recent events</h3>
    <div id="rows" class="scrollbox"></div>
  </div>
</div>

<div id="modal"><div id="card">
  <div style="font-size:13px;color:#aaa;margin-bottom:8px">Add this person</div>
  <img id="cardImg" src="">
  <input id="nameInput" placeholder="Name">
  <input id="companyInput" placeholder="Company" list="companyList">
  <datalist id="companyList"></datalist>
  <div class="btns">
    <button onclick="closeModal()">Cancel</button>
    <button class="primary" onclick="assignPending()">Assign</button>
  </div>
</div></div>

<div id="attModal"><div id="attCard">
  <div style="font-size:16px;font-weight:bold;margin-bottom:12px">🕐 Attendance</div>
  <label class="attRow"><input type="checkbox" id="attEnabled"> Attendance recording on</label>
  <div class="attGrid">
    <div>Morning in — from</div><input type="time" id="mStart">
    <div>to</div><input type="time" id="mEnd">
    <div>Evening out — from</div><input type="time" id="eStart">
    <div>to</div><input type="time" id="eEnd">
  </div>
  <div style="font-size:12px;color:#888;margin:6px 0 12px">Faces are only recorded for attendance during these windows. Each person is logged once (check-in) in the morning and their last exit (check-out) in the evening.</div>
  <div class="btns"><button onclick="closeAtt()">Close</button><button class="primary" onclick="saveAtt()">Save</button></div>
  <div style="font-weight:bold;margin:16px 0 6px">Today</div>
  <div id="attToday" class="scrollbox" style="max-height:220px"></div>
</div></div>

<script>
const vid = document.getElementById('vid'), cv = document.getElementById('overlay'), ctx = cv.getContext('2d');
const colors = {person: '#ff8c00', known: '#4caf50', unknown: '#e53935'};

function resizeCanvas(){
  const r = vid.getBoundingClientRect();
  cv.width = r.width; cv.height = r.height;
}
window.addEventListener('resize', resizeCanvas);
vid.addEventListener('load', resizeCanvas);
setTimeout(resizeCanvas, 300);

let lastTs = 0, updates = 0, fpsWindowStart = Date.now();
async function pollDetections(){
  try {
    const r = await fetch('/detections'); const d = await r.json();
    if (d.frame_w && cv.width > 0) {
      resizeCanvas();
      const sx = cv.width / d.frame_w, sy = cv.height / d.frame_h;
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.font = 'bold 22px sans-serif'; ctx.lineWidth = 3;
      for (const det of d.detections) {
        const [x, y, w, h] = det.box;
        const color = colors[det.kind] || colors[det.type] || '#fff';
        if (!det.point_only) {
          ctx.strokeStyle = color;
          ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
        }
        const ty = Math.max(20, y * sy - 6);
        ctx.lineWidth = 4; ctx.strokeStyle = '#000';
        ctx.strokeText(det.label, x * sx, ty);   // dark outline so text reads on any background
        ctx.fillStyle = color;
        ctx.fillText(det.label, x * sx, ty);
        ctx.lineWidth = 3;
      }
      if (d.ts !== lastTs) { lastTs = d.ts; updates++; }
    }
  } catch (e) {}
  setTimeout(pollDetections, 100);
}
pollDetections();

setInterval(() => {
  const now = Date.now();
  document.getElementById('fps').textContent = `detections: ${(updates / ((now - fpsWindowStart) / 1000)).toFixed(1)}/s`;
  updates = 0; fpsWindowStart = now;
}, 2000);

async function pollEvents(){
  const r = await fetch('/events'); const rows = await r.json();
  document.getElementById('rows').innerHTML = rows.map(e =>
    `<div class="row ${e.kind}">${e.ts}${e.location ? ' · ' + e.location : ''}<br><b>${e.name}</b> (${e.kind})${e.company ? ' — ' + e.company : ''}</div>`).join('');
}
setInterval(pollEvents, 3000); pollEvents();

// ---- location bar ----
async function loadSettings(){
  const r = await fetch('/settings'); const s = await r.json();
  document.getElementById('locInput').value = s.location || '';
  document.getElementById('companyInput').value = s.default_company || '';
}
async function saveLocation(){
  await fetch('/settings', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({location: document.getElementById('locInput').value})});
}
loadSettings();

// ---- rotate toggle ----
async function toggleRotate(){
  await fetch('/rotate', {method:'POST'});
}

// ---- unknown-face list (sidebar, passive -- never pops up on its own) ----
let currentPendingId = null;
async function pollPending(){
  try {
    const r = await fetch('/pending'); const list = await r.json();
    const box = document.getElementById('pendingList');
    box.innerHTML = list.length === 0
      ? '<span style="font-size:12px;color:#666">none right now</span>'
      : list.map(p => `<div class="pcard ${p.acknowledged ? '' : 'alerting'}">
           <img src="/pending_crop/${p.id}">
           <span><b>${p.label}</b><br>${p.camera}</span>
           ${p.acknowledged ? '' : `<button onclick="acknowledgePending('${p.id}')">Acknowledged</button>`}
           <button onclick="openAssign('${p.id}')">Add</button>
         </div>`).join('');
  } catch (e) {}
  setTimeout(pollPending, 1500);
}
pollPending();

async function acknowledgePending(pendingId){
  await fetch('/acknowledge', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pending_id: pendingId})});
}

async function openAssign(pendingId){
  currentPendingId = pendingId;
  document.getElementById('cardImg').src = '/pending_crop/' + pendingId + '?t=' + Date.now();
  document.getElementById('nameInput').value = '';
  const compR = await fetch('/companies'); const comps = await compR.json();
  document.getElementById('companyList').innerHTML = comps.map(c => `<option value="${c}">`).join('');
  const settR = await fetch('/settings'); const sett = await settR.json();
  document.getElementById('companyInput').value = sett.default_company || '';
  document.getElementById('modal').style.display = 'flex';
}
function closeModal(){
  document.getElementById('modal').style.display = 'none';
  currentPendingId = null;
}
async function assignPending(){
  const name = document.getElementById('nameInput').value.trim();
  if (!name) { alert('Enter a name'); return; }
  const company = document.getElementById('companyInput').value.trim();
  await fetch('/assign', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({pending_id: currentPendingId, name, company})});
  closeModal();
}

// ---- attendance ----
const PHASE_LABEL = {morning: 'CHECK-IN OPEN', evening: 'CHECK-OUT OPEN'};
async function pollAttStatus(){
  try {
    const r = await fetch('/attendance_status'); const s = await r.json();
    const pill = document.getElementById('attPill');
    if (!s.enabled) { pill.textContent = 'Attendance: off'; pill.classList.remove('open'); }
    else if (s.open) { pill.textContent = PHASE_LABEL[s.phase]; pill.classList.add('open'); }
    else { pill.textContent = 'Attendance: waiting'; pill.classList.remove('open'); }
  } catch (e) {}
  setTimeout(pollAttStatus, 5000);
}
pollAttStatus();

function fmtTime(iso){ return iso ? iso.slice(11,16) : '—'; }
async function openAttendance(){
  const c = await (await fetch('/attendance_config')).json();
  document.getElementById('attEnabled').checked = c.enabled;
  document.getElementById('mStart').value = c.morning_start;
  document.getElementById('mEnd').value = c.morning_end;
  document.getElementById('eStart').value = c.evening_start;
  document.getElementById('eEnd').value = c.evening_end;
  const list = await (await fetch('/attendance')).json();
  document.getElementById('attToday').innerHTML = list.length === 0
    ? '<div style="color:#666;font-size:13px;padding:8px">No one recorded yet today</div>'
    : list.map(a => `<div class="attItem"><span class="nm">${a.name || '—'}</span>
        <span><span class="tm">in ${fmtTime(a.check_in)}</span> &nbsp; <span class="tm out">out ${fmtTime(a.check_out)}</span></span></div>`).join('');
  document.getElementById('attModal').style.display = 'flex';
}
function closeAtt(){ document.getElementById('attModal').style.display = 'none'; }
async function saveAtt(){
  const body = {
    enabled: document.getElementById('attEnabled').checked,
    morning_start: document.getElementById('mStart').value,
    morning_end: document.getElementById('mEnd').value,
    evening_start: document.getElementById('eStart').value,
    evening_end: document.getElementById('eEnd').value,
  };
  await fetch('/attendance_config', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  closeAtt();
  pollAttStatus();
}
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/raw_feed")
def raw_feed():
    """Undecorated video, as fast as new camera frames arrive -- this is the smooth stream."""
    def gen():
        while True:
            jpeg = camera.get_raw_jpeg()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections():
    """Latest YOLO+face results as plain coordinates -- browser overlays these on the video."""
    return jsonify(camera.get_detections())


@app.route("/events")
def events():
    return jsonify(recent_events())


@app.route("/snapshots/<path:fn>")
def snapshot(fn):
    return send_from_directory(SNAP_DIR, fn)


@app.route("/pending")
def pending():
    return jsonify(camera.get_pending_list())


@app.route("/pending_crop/<pid>")
def pending_crop(pid):
    crop = camera.get_pending_crop(pid)
    if crop is None:
        return "", 404
    return Response(crop, mimetype="image/jpeg")


@app.route("/assign", methods=["POST"])
def assign():
    body = request.get_json(force=True)
    result = camera.assign_pending(body.get("pending_id"), body.get("name", "").strip(), body.get("company", "").strip())
    if result is None:
        return jsonify({"ok": False, "error": "pending id not found (may have expired)"}), 404
    return jsonify({"ok": True, **result})


@app.route("/skip", methods=["POST"])
def skip():
    body = request.get_json(force=True)
    ok = camera.skip_pending(body.get("pending_id"))
    return jsonify({"ok": ok})


@app.route("/acknowledge", methods=["POST"])
def acknowledge():
    body = request.get_json(force=True)
    ok = camera.acknowledge_pending(body.get("pending_id"))
    return jsonify({"ok": ok})


@app.route("/companies")
def companies():
    con = db()
    rows = [r[0] for r in con.execute("SELECT name FROM companies ORDER BY name")]
    con.close()
    return jsonify(rows)


@app.route("/settings", methods=["GET", "POST"])
def settings():
    con = db()
    if request.method == "POST":
        body = request.get_json(force=True)
        if "location" in body:
            set_setting(con, "location", body["location"])
    result = {"location": get_setting(con, "location", ""), "default_company": get_setting(con, "default_company", "")}
    con.close()
    return jsonify(result)


@app.route("/people")
def people():
    con = db()
    rows = con.execute("""SELECT p.id, p.name, c.name, p.created_at FROM people p
                           LEFT JOIN companies c ON c.id = p.company_id ORDER BY p.created_at DESC""").fetchall()
    con.close()
    return jsonify([{"id": r[0], "name": r[1], "company": r[2], "created_at": r[3]} for r in rows])


@app.route("/status")
def status():
    return jsonify({"camera_status": camera.status, "known_people": [e["name"] for e in known_gallery.values()]})


@app.route("/rotate", methods=["GET", "POST"])
def rotate():
    if request.method == "POST":
        camera.rotate_180 = not camera.rotate_180
    return jsonify({"rotate_180": camera.rotate_180})


@app.route("/attendance")
def attendance():
    date = request.args.get("date")  # YYYY-MM-DD, defaults to today
    return jsonify(attendance_for_date(date))


@app.route("/attendance_status")
def attendance_status():
    cfg = get_attendance_config()
    phase = current_attendance_phase(cfg)
    return jsonify({"enabled": cfg["enabled"], "phase": phase, "open": phase is not None,
                    "now": datetime.now().strftime("%H:%M")})


@app.route("/attendance_config", methods=["GET", "POST"])
def attendance_config():
    con = db()
    if request.method == "POST":
        body = request.get_json(force=True)
        set_setting(con, "att_enabled", "1" if body.get("enabled") else "0")
        for field, key in (("morning_start", "att_morning_start"), ("morning_end", "att_morning_end"),
                           ("evening_start", "att_evening_start"), ("evening_end", "att_evening_end")):
            val = (body.get(field) or "").strip()
            if _valid_hhmm(val):
                set_setting(con, key, val)
    cfg = get_attendance_config(con)
    con.close()
    return jsonify(cfg)


def _valid_hhmm(s):
    """Guard against bad time input silently corrupting a window (which would disable attendance)."""
    if not s or len(s) != 5 or s[2] != ":":
        return False
    try:
        hh, mm = int(s[:2]), int(s[3:])
    except ValueError:
        return False
    return 0 <= hh <= 23 and 0 <= mm <= 59


@app.route("/db_data")
def db_data():
    """Everything needed to render the /database explorer page in one call."""
    con = db()
    companies = con.execute("""SELECT c.id, c.name, c.created_at, COUNT(p.id)
                                FROM companies c LEFT JOIN people p ON p.company_id = c.id
                                GROUP BY c.id ORDER BY c.name""").fetchall()
    people_rows = con.execute("""SELECT p.id, p.name, c.name, p.created_at, COUNT(e.id)
                                  FROM people p LEFT JOIN companies c ON c.id = p.company_id
                                  LEFT JOIN face_embeddings e ON e.person_id = p.id
                                  GROUP BY p.id ORDER BY p.created_at DESC""").fetchall()
    event_counts = dict(con.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
    total_events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    products = con.execute("SELECT id, sku, name, description, created_at FROM products ORDER BY id").fetchall()
    stock_events = con.execute("""SELECT se.id, se.ts, pr.name, se.action, se.quantity, pe.name, se.location
                                   FROM stock_events se LEFT JOIN products pr ON pr.id = se.product_id
                                   LEFT JOIN people pe ON pe.id = se.person_id
                                   ORDER BY se.id DESC LIMIT 200""").fetchall()
    con.close()
    today_attendance = attendance_for_date()
    unknown_snaps = sorted(
        (f for f in os.listdir(SNAP_DIR) if f.lower().startswith("unknown")),
        reverse=True)[:200]
    return jsonify({
        "attendance_today": today_attendance,
        "companies": [{"id": r[0], "name": r[1], "created_at": r[2], "people_count": r[3]} for r in companies],
        "people": [{"id": r[0], "name": r[1], "company": r[2], "created_at": r[3], "embeddings": r[4]}
                   for r in people_rows],
        "event_counts": event_counts,
        "total_events": total_events,
        "unknown_snapshot_count": len(os.listdir(SNAP_DIR)) and
                                   sum(1 for f in os.listdir(SNAP_DIR) if f.lower().startswith("unknown")),
        "unknown_snapshots": unknown_snaps,
        "products": [{"id": r[0], "sku": r[1], "name": r[2], "description": r[3], "created_at": r[4]}
                     for r in products],
        "stock_events": [{"id": r[0], "ts": r[1], "product": r[2], "action": r[3], "quantity": r[4],
                          "person": r[5], "location": r[6]} for r in stock_events],
    })


DB_PAGE = """<!doctype html><html><head><title>camera_ai — database</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{background:#0b0b0b;color:#eee;font-family:sans-serif;margin:0;padding:24px}
h1{margin-top:0} h2{border-bottom:1px solid #333;padding-bottom:6px;margin-top:32px}
a.back{color:#4caf50;text-decoration:none;font-size:14px}
table{width:100%;border-collapse:collapse;margin-top:8px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #222;font-size:14px}
th{color:#999;font-weight:normal}
.scrollbox{max-height:360px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:#555 #1a1a1a;border:1px solid #222}
.scrollbox::-webkit-scrollbar{width:8px}
.scrollbox::-webkit-scrollbar-track{background:#1a1a1a}
.scrollbox::-webkit-scrollbar-thumb{background:#555;border-radius:4px}
.badge{display:inline-block;background:#222;border-radius:10px;padding:2px 10px;font-size:13px;margin-right:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;padding:10px}
.grid img{width:100%;border-radius:4px;border:1px solid #333}
.grid div{font-size:11px;color:#888;text-align:center;margin-top:2px;word-break:break-all}
.empty{color:#666;font-size:13px;padding:10px}
</style></head><body>
<a class="back" href="/">&larr; back to live view</a>
<h1>Database explorer</h1>
<div id="summary"></div>

<h2>Today's attendance</h2>
<div id="attendanceToday" class="scrollbox"></div>

<h2>Companies</h2>
<div id="companies" class="scrollbox"></div>

<h2>People (enrolled faces)</h2>
<div id="people" class="scrollbox"></div>

<h2>Unknown-face snapshots</h2>
<div id="unknownGrid" class="scrollbox grid"></div>

<h2>Products <span style="font-size:13px;color:#666;font-weight:normal">(inventory pipeline -- schema ready, not built yet)</span></h2>
<div id="products" class="scrollbox"></div>

<h2>Stock events <span style="font-size:13px;color:#666;font-weight:normal">(who took/added what -- schema ready, not built yet)</span></h2>
<div id="stockEvents" class="scrollbox"></div>

<script>
async function load(){
  const r = await fetch('/db_data'); const d = await r.json();

  document.getElementById('summary').innerHTML =
    `<span class="badge">${d.companies.length} companies</span>` +
    `<span class="badge">${d.people.length} enrolled people</span>` +
    `<span class="badge">${d.total_events} total events</span>` +
    `<span class="badge">${d.event_counts.known || 0} known sightings</span>` +
    `<span class="badge">${d.event_counts.unknown || 0} unknown sightings</span>` +
    `<span class="badge">${d.unknown_snapshot_count} unknown photos stored</span>` +
    `<span class="badge">${d.attendance_today.length} present today</span>`;

  const att = d.attendance_today;
  document.getElementById('attendanceToday').innerHTML = att.length === 0
    ? '<div class="empty">no one recorded yet today</div>'
    : '<table><tr><th>Name</th><th>Company</th><th>Check-in</th><th>Check-out</th></tr>' +
      att.map(a => `<tr><td>${a.name || '—'}</td><td>${a.company || '—'}</td>` +
        `<td>${a.check_in ? a.check_in.slice(11,16) : '—'}</td>` +
        `<td>${a.check_out ? a.check_out.slice(11,16) : '—'}</td></tr>`).join('') +
      '</table>';

  document.getElementById('companies').innerHTML = d.companies.length === 0
    ? '<div class="empty">none yet</div>'
    : '<table><tr><th>Name</th><th>People</th><th>Created</th></tr>' +
      d.companies.map(c => `<tr><td>${c.name}</td><td>${c.people_count}</td><td>${c.created_at || ''}</td></tr>`).join('') +
      '</table>';

  document.getElementById('people').innerHTML = d.people.length === 0
    ? '<div class="empty">none yet</div>'
    : '<table><tr><th>Name</th><th>Company</th><th>Reference photos</th><th>Enrolled</th></tr>' +
      d.people.map(p => `<tr><td>${p.name}</td><td>${p.company || '—'}</td><td>${p.embeddings}</td><td>${p.created_at || ''}</td></tr>`).join('') +
      '</table>';

  document.getElementById('unknownGrid').innerHTML = d.unknown_snapshots.length === 0
    ? '<div class="empty">none yet</div>'
    : d.unknown_snapshots.map(f => `<div><img src="/snapshots/${f}" loading="lazy"><div>${f}</div></div>`).join('');

  document.getElementById('products').innerHTML = d.products.length === 0
    ? '<div class="empty">no products tracked yet</div>'
    : '<table><tr><th>SKU</th><th>Name</th><th>Description</th></tr>' +
      d.products.map(p => `<tr><td>${p.sku || ''}</td><td>${p.name}</td><td>${p.description || ''}</td></tr>`).join('') +
      '</table>';

  document.getElementById('stockEvents').innerHTML = d.stock_events.length === 0
    ? '<div class="empty">no stock events logged yet</div>'
    : '<table><tr><th>Time</th><th>Product</th><th>Action</th><th>Qty</th><th>Person</th></tr>' +
      d.stock_events.map(s => `<tr><td>${s.ts}</td><td>${s.product || ''}</td><td>${s.action || ''}</td><td>${s.quantity ?? ''}</td><td>${s.person || ''}</td></tr>`).join('') +
      '</table>';
}
load();
</script>
</body></html>"""


@app.route("/database")
def database_page():
    return DB_PAGE


if __name__ == "__main__":
    print(f"[web] open http://localhost:{HTTP_PORT}", flush=True)
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)
