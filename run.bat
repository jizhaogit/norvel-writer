@echo off
setlocal enabledelayedexpansion

:: Norvel Writer - Quick Launcher
:: Double-click this file to install dependencies and start the app.

echo ============================================================
echo  Norvel Writer - Local Writing Assistant
echo ============================================================
echo.

:: Detect Python — try 'python', then 'py' (Windows launcher), then 'python3'
set PYTHON=
python --version >nul 2>&1 && set PYTHON=python
if "!PYTHON!"=="" (
    py --version >nul 2>&1 && set PYTHON=py
)
if "!PYTHON!"=="" (
    python3 --version >nul 2>&1 && set PYTHON=python3
)

if "!PYTHON!"=="" (
    echo ERROR: Python not found.
    echo Please install Python 3.11 or newer from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    echo If Python is already installed, try running this from a Command Prompt:
    echo   python -m norvel_writer
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('!PYTHON! --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER% ^(!PYTHON!^)

:: Install dependencies (only if not already installed)
!PYTHON! -c "import norvel_writer" >nul 2>&1
if errorlevel 1 (
    echo Installing Norvel Writer dependencies... ^(first run only, may take a few minutes^)
    echo.
    !PYTHON! -m pip install -e "%~dp0." --quiet
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo Try running manually: !PYTHON! -m pip install -e .
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
    echo.
)

:: Launch the app
echo Starting Norvel Writer...
!PYTHON! -m norvel_writer

if errorlevel 1 (
    echo.
    echo App exited with an error. Check logs in %APPDATA%\NorvelWriter\logs\
    pause
)
