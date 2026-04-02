@echo off
setlocal enabledelayedexpansion

:: Norvel Writer - Quick Launcher
:: Double-click this file to install dependencies and start the app.

echo ============================================================
echo  Norvel Writer - Local Writing Assistant
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Please install Python 3.11 or newer from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

:: Install dependencies (only if not already installed)
python -c "import norvel_writer" >nul 2>&1
if errorlevel 1 (
    echo Installing Norvel Writer dependencies... (first run only, may take a few minutes)
    echo.
    pip install -e "%~dp0." --quiet
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo Try running: pip install -e .
        pause
        exit /b 1
    )
    echo Dependencies installed successfully.
    echo.
)

:: Launch the app
echo Starting Norvel Writer...
python -m norvel_writer

if errorlevel 1 (
    echo.
    echo App exited with an error. Check logs in %%APPDATA%%\NorvelWriter\logs\
    pause
)
