# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Norvel Writer (Windows, --onedir)."""

import sys
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"

a = Analysis(
    [str(SRC / "norvel_writer" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        # Bundle prompt templates
        (str(SRC / "norvel_writer" / "resources"), "norvel_writer/resources"),
        # NLTK punkt data (pre-downloaded by build script)
        ("nltk_data", "nltk_data"),
    ],
    hiddenimports=[
        "chromadb",
        "chromadb.api",
        "chromadb.api.client",
        "chromadb.db.impl.sqlite",
        "hnswlib",
        "ollama",
        "pdfplumber",
        "docx",
        "langdetect",
        "jinja2",
        "watchdog",
        "watchdog.observers",
        "watchdog.observers.polling",
        "pydantic",
        "pydantic_settings",
        "tomli_w",
        "platformdirs",
        "nltk",
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "numpy", "scipy"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NorvelWriter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add .ico path here when available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NorvelWriter",
)
