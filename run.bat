@echo off
REM ============================================================
REM  FaceFoundry — one-click launcher (Windows)
REM  Starts the local control panel and opens it in your browser.
REM ============================================================
setlocal
cd /d "%~dp0"

echo [FaceFoundry] Checking dependencies...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo [FaceFoundry] ERROR: could not install requirements. Is Python on PATH?
  pause
  exit /b 1
)

if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
  echo [FaceFoundry] WARNING: %USERPROFILE%\.kaggle\kaggle.json not found.
  echo               Jobs will fail until you add your Kaggle API token. See SETUP.md.
)

echo [FaceFoundry] Starting control panel at http://localhost:8000
start "" http://localhost:8000
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause
