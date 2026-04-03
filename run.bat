@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Norvel Writer
echo ============================================================
echo.

set "UV="

where uv >nul 2>&1
if not errorlevel 1 set "UV=uv"

if "!UV!"=="" (
    if exist "%APPDATA%\uv\bin\uv.exe" set "UV=%APPDATA%\uv\bin\uv.exe"
)
if "!UV!"=="" (
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if "!UV!"=="" (
    if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV=%USERPROFILE%\.cargo\bin\uv.exe"
)

if not "!UV!"=="" goto :launch

echo uv not found. Downloading uv ^(one-time setup, ~15 MB^)...
echo uv manages Python automatically - no separate Python install needed.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%APPDATA%\uv\bin\uv.exe" set "UV=%APPDATA%\uv\bin\uv.exe"
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"

if "!UV!"=="" (
    echo.
    echo ERROR: Could not install uv automatically.
    echo Please install it from: https://astral.sh/uv
    pause
    exit /b 1
)

:launch
cd /d "%~dp0"

echo Starting Norvel Writer...
echo ^(First run installs Python 3.11 + packages - may take a few minutes^)
echo.

rem Suppress uv "hardlink → copy" warning that appears when the cache and
rem project sit on different drives/filesystems.
set "UV_LINK_MODE=copy"

"!UV!" run --python 3.11 python -m norvel_writer

if errorlevel 1 (
    echo.
    echo App exited with an error.
    echo Check logs in: %APPDATA%\NorvelWriter\logs\
    pause
)
