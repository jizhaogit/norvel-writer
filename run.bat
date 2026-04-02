@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Norvel Writer - Local Writing Assistant
echo ============================================================
echo.

:: ── Find uv ───────────────────────────────────────────────────────────────
:: uv manages Python automatically — no separate Python install required.
:: https://astral.sh/uv

set "UV="

:: Check PATH first
where uv >nul 2>&1 && set "UV=uv"

:: Common install locations on Windows
if "!UV!"=="" if exist "%APPDATA%\uv\bin\uv.exe"        set "UV=%APPDATA%\uv\bin\uv.exe"
if "!UV!"=="" if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if "!UV!"=="" if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV=%USERPROFILE%\.cargo\bin\uv.exe"

:: If still not found, download and install uv automatically
if "!UV!"=="" (
    echo uv not found. Downloading uv ^(one-time setup, ~15 MB^)...
    echo uv manages Python automatically so you do not need to install Python separately.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    :: Re-check after install
    if exist "%APPDATA%\uv\bin\uv.exe"        set "UV=%APPDATA%\uv\bin\uv.exe"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)

if "!UV!"=="" (
    echo.
    echo ERROR: Could not install uv automatically.
    echo Please install it manually from: https://astral.sh/uv
    echo   Or install Python 3.11+ from: https://python.org
    pause
    exit /b 1
)

echo.

:: ── Launch app ────────────────────────────────────────────────────────────
:: uv will auto-install Python 3.11 and all dependencies on first run.
:: Subsequent runs skip already-installed packages and start in seconds.
cd /d "%~dp0"

echo Starting Norvel Writer...
echo ^(First run installs Python 3.11 + dependencies — this may take a few minutes^)
echo.

"!UV!" run --python 3.11 python -m norvel_writer

if errorlevel 1 (
    echo.
    echo App exited with an error. Check logs in %APPDATA%\NorvelWriter\logs\
    pause
)
