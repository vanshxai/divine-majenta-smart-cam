# PyInstaller spec -- builds a standalone Windows app (one folder).
# Usage (on the Windows machine, inside the Python env):  pyinstaller camera_ai.spec --noconfirm
# Produces: dist/DivineSmartCam/DivineSmartCam.exe
#
# Model files (yunet.onnx, sface.onnx, yolov8n.pt) must be present next to this spec BEFORE building
# -- they are gitignored, so download them first (see PACKAGING.md). They get baked into the bundle.
import os
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ["cv2"]

# bake in the model weights sitting next to this spec
for _m in ("yunet.onnx", "sface.onnx", "yolov8n.pt"):
    if os.path.exists(_m):
        datas.append((_m, "."))

# ultralytics + torch ship data files / submodules PyInstaller can't infer on its own
for _pkg in ("ultralytics", "torch", "torchvision"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception as _e:
        print(f"[spec] collect_all({_pkg}) skipped: {_e}")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["mediapipe", "matplotlib", "tkinter"],  # not used -- keeps the build smaller
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DivineSmartCam",
    debug=False,
    strip=False,
    upx=False,
    console=True,   # keep the console so startup/[camera] logs are visible; set False for a silent app
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DivineSmartCam",
)
