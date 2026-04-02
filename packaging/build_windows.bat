@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Norvel Writer - Windows Build Script
echo ============================================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install -e ".[dev]" --quiet
if errorlevel 1 (
    echo ERROR: pip install failed
    exit /b 1
)

:: Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

:: Clean previous build
if exist dist\NorvelWriter rmdir /s /q dist\NorvelWriter
if exist build rmdir /s /q build

:: Run PyInstaller
echo Building with PyInstaller...
pyinstaller packaging\norvel_writer.spec --noconfirm

if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

echo.
echo ============================================================
echo  Build complete: dist\NorvelWriter\NorvelWriter.exe
echo ============================================================
