@echo off
REM ============================================================================
REM  Build the Divine Smart Cam Windows app.
REM  Run this from the repo folder, inside your Python environment (see PACKAGING.md).
REM ============================================================================
setlocal

echo [1/4] Checking model files...
if not exist yunet.onnx  ( echo   MISSING yunet.onnx  -- see PACKAGING.md & goto :fail )
if not exist sface.onnx  ( echo   MISSING sface.onnx  -- see PACKAGING.md & goto :fail )
if not exist yolov8n.pt  ( echo   MISSING yolov8n.pt  -- see PACKAGING.md & goto :fail )
echo   OK

echo [2/4] Checking ffmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  if not exist ffmpeg\ffmpeg.exe echo   WARNING: ffmpeg not on PATH and no ffmpeg\ffmpeg.exe found.
  if not exist ffmpeg\ffmpeg.exe echo            The app needs ffmpeg at run time -- see PACKAGING.md.
) else ( echo   OK )

echo [3/4] Installing PyInstaller...
pip install pyinstaller || goto :fail

echo [4/4] Building (this takes a few minutes)...
pyinstaller camera_ai.spec --noconfirm || goto :fail

echo.
echo ============================================================================
echo  DONE.  App: dist\DivineSmartCam\DivineSmartCam.exe
echo  Before running, put these NEXT TO that .exe:
echo    - config.json        (copy config.example.json, set your camera_ip)
echo    - .camera_user       (camera username, one line)
echo    - .camera_pw         (camera password, one line)
echo    - ffmpeg\ folder     (ffmpeg.exe + ffprobe.exe) unless ffmpeg is on PATH
echo  Then run the .exe and open http://localhost:5055
echo ============================================================================
goto :eof

:fail
echo.
echo BUILD FAILED -- see the message above.
exit /b 1
