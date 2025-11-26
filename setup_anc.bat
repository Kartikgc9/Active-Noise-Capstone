@echo off
REM Setup script for ANC System (Windows)

echo =======================================================================
echo Active Noise Cancellation System - Setup (Windows)
echo =======================================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

REM Ask about virtual environment
set /p create_venv="Create a virtual environment? (recommended) [y/n]: "

if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv anc_env
    call anc_env\Scripts\activate.bat
    echo Virtual environment activated!
    echo.
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing ANC system requirements...
pip install -r requirements_anc.txt
echo.

REM Test installation
echo Testing installation...
python -c "import numpy, scipy, librosa, soundfile, sounddevice; print('All core libraries installed successfully!')" 2>nul
if errorlevel 1 (
    echo ERROR: Installation test failed. Please check errors above.
    pause
    exit /b 1
)
echo.

REM Show audio devices
echo Available audio devices:
python -c "import sounddevice as sd; print(sd.query_devices())"
echo.

echo =======================================================================
echo Setup Complete!
echo =======================================================================
echo.
echo To run the ANC system:
echo   python scripts\anc_system.py gentle
echo.
echo To test the system:
echo   python scripts\test_anc_system.py
echo.
if /i "%create_venv%"=="y" (
    echo Note: Virtual environment created. To activate it later:
    echo   anc_env\Scripts\activate
    echo.
)
echo =======================================================================
pause
