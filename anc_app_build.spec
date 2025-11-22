# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Advanced ANC System GUI Application
#
# Build instructions:
#   1. Install PyInstaller: pip install pyinstaller
#   2. Run: pyinstaller anc_app_build.spec
#   3. Find executable in: dist/AdvancedANCSystem/

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files and hidden imports
hiddenimports = [
    'scipy',
    'scipy.signal',
    'scipy.special',
    'scipy.sparse',
    'scipy.ndimage',
    'numpy',
    'librosa',
    'soundfile',
    'sounddevice',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
]

# Collect data files for librosa (includes audio samples and data)
datas = []
datas += collect_data_files('librosa')
datas += collect_data_files('soundfile')

a = Analysis(
    ['scripts/anc_gui_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Exclude if not needed
        'PIL',
        'tkinter',
    ],
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
    name='AdvancedANCSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for GUI app (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path here if you have one: 'path/to/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AdvancedANCSystem',
)
