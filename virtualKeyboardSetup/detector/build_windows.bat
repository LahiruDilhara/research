@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Virtual Keyboard Detector - Windows Build Script
echo  Uses: uv + PyInstaller
echo ============================================================
echo.

REM ── 1. Check Python ──────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.12 from https://python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo [INFO] Found %%v

REM ── 2. Install uv if missing ─────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing uv package manager...
    pip install uv --quiet
)
echo [INFO] uv ready.

REM ── 3. Create and sync virtual environment ───────────────────
echo [INFO] Creating virtual environment and installing dependencies...
uv sync
if errorlevel 1 (
    echo [ERROR] uv sync failed. Check your internet connection and pyproject.toml.
    pause
    exit /b 1
)
echo [INFO] Dependencies installed.

REM ── 4. Install PyInstaller into the venv ─────────────────────
echo [INFO] Installing PyInstaller...
uv pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] PyInstaller install failed.
    pause
    exit /b 1
)

REM ── 5. Run PyInstaller ───────────────────────────────────────
echo [INFO] Building Windows executable...
echo.

uv run pyinstaller ^
    --name "VirtualKeyboardDetector" ^
    --onedir ^
    --windowed ^
    --add-data "hand_landmarker.task;." ^
    --add-data "ai_model_plugins;ai_model_plugins" ^
    --add-data ".env.example;." ^
    --collect-all "qfluentwidgets" ^
    --collect-all "mediapipe" ^
    --collect-all "dt_apriltags" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    --hidden-import "sklearn" ^
    --hidden-import "sklearn.utils._cython_blas" ^
    --hidden-import "sklearn.neighbors._typedefs" ^
    --hidden-import "sklearn.tree._utils" ^
    --hidden-import "pynput.keyboard._win32" ^
    --hidden-import "pynput.mouse._win32" ^
    --noconfirm ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Output folder: dist\VirtualKeyboardDetector\
echo  Run: dist\VirtualKeyboardDetector\VirtualKeyboardDetector.exe
echo ============================================================
echo.
pause
