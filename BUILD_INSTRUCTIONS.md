# Building Advanced ANC System as Executable

This guide explains how to build the Advanced ANC System as a standalone `.exe` file that can run on any Windows computer without requiring Python installation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Building the Executable](#building-the-executable)
4. [Distribution](#distribution)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.8 or higher** (3.9-3.11 recommended)
- **pip** (Python package installer)
- **Git** (optional, for cloning repository)

### System Requirements

- **Windows 10/11** (for .exe builds)
- **4 GB RAM** minimum (8 GB recommended)
- **500 MB free disk space**
- **Microphone and speakers/headphones**

---

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/yourusername/Active-Noise-Capstone.git
cd Active-Noise-Capstone
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install GUI application requirements
pip install -r requirements_gui.txt

# This will also install PyInstaller for building the executable
```

---

## Building the Executable

### Method 1: Using PyInstaller Spec File (Recommended)

The project includes a pre-configured PyInstaller spec file (`anc_app_build.spec`) that handles all dependencies automatically.

```bash
# Build the executable
pyinstaller anc_app_build.spec

# The executable will be created in: dist/AdvancedANCSystem/
```

### Method 2: Using PyInstaller Command Line

If you prefer to use command-line options:

```bash
pyinstaller --name="AdvancedANCSystem" \
            --windowed \
            --onedir \
            --hidden-import=scipy \
            --hidden-import=librosa \
            --hidden-import=sounddevice \
            --hidden-import=PyQt5 \
            scripts/anc_gui_app.py
```

### Method 3: One-File Executable (Slower Startup)

To create a single .exe file instead of a folder:

```bash
pyinstaller --name="AdvancedANCSystem" \
            --windowed \
            --onefile \
            --hidden-import=scipy \
            --hidden-import=librosa \
            --hidden-import=sounddevice \
            --hidden-import=PyQt5 \
            scripts/anc_gui_app.py
```

**Note**: One-file mode is simpler to distribute but has slower startup time.

---

## Build Output

After building, you'll find:

```
dist/
└── AdvancedANCSystem/
    ├── AdvancedANCSystem.exe    # Main executable
    ├── _internal/                # Required libraries and dependencies
    └── [various DLL files]       # Python runtime and libraries
```

### Running the Application

1. Navigate to `dist/AdvancedANCSystem/`
2. Double-click `AdvancedANCSystem.exe`
3. The GUI application will launch

---

## Distribution

### Creating a Distributable Package

#### Option 1: ZIP Archive

```bash
# Navigate to dist folder
cd dist

# Create ZIP file (use Windows Explorer or command line)
# Windows PowerShell:
Compress-Archive -Path AdvancedANCSystem -DestinationPath AdvancedANCSystem.zip

# Or use 7-Zip, WinRAR, etc.
```

#### Option 2: Installer (Advanced)

For a professional installer, use tools like:

- **Inno Setup** (free, open-source): https://jrsoftware.org/isinfo.php
- **NSIS** (free): https://nsis.sourceforge.io/
- **InstallForge** (free): https://installforge.net/

Example Inno Setup script (`installer.iss`):

```iss
[Setup]
AppName=Advanced ANC System
AppVersion=1.0
DefaultDirName={pf}\AdvancedANCSystem
DefaultGroupName=Advanced ANC System
OutputDir=installer_output
OutputBaseFilename=AdvancedANCSystem_Setup

[Files]
Source: "dist\AdvancedANCSystem\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\Advanced ANC System"; Filename: "{app}\AdvancedANCSystem.exe"
Name: "{commondesktop}\Advanced ANC System"; Filename: "{app}\AdvancedANCSystem.exe"
```

### Size Optimization

The default build is ~200-400 MB. To reduce size:

1. **Use UPX compression** (already enabled in spec file):
   ```bash
   # Install UPX
   # Download from: https://upx.github.io/
   # Place upx.exe in PATH or PyInstaller directory
   ```

2. **Exclude unnecessary libraries** in the spec file:
   ```python
   excludes=['matplotlib', 'PIL', 'tkinter', 'pandas']
   ```

3. **Use one-file mode** (simpler but slower startup)

---

## Cross-Platform Building

### Building for Linux

```bash
# On Linux system
pip install -r requirements_gui.txt
pyinstaller anc_app_build.spec

# Output: dist/AdvancedANCSystem/AdvancedANCSystem (no .exe extension)
```

### Building for macOS

```bash
# On macOS
pip install -r requirements_gui.txt
pyinstaller anc_app_build.spec

# Output: dist/AdvancedANCSystem/AdvancedANCSystem
# Or create .app bundle with --windowed
```

**Note**: You must build on the target platform. Cross-compilation is not supported by PyInstaller.

---

## Troubleshooting

### Common Issues

#### 1. Missing DLL Errors

**Error**: "DLL load failed" or "Module not found"

**Solution**:
```bash
# Reinstall dependencies
pip uninstall -y librosa soundfile sounddevice scipy numpy
pip install librosa soundfile sounddevice scipy numpy --no-cache-dir

# Rebuild
pyinstaller --clean anc_app_build.spec
```

#### 2. Import Errors

**Error**: "No module named 'scipy.signal'"

**Solution**: Add to `hiddenimports` in spec file:
```python
hiddenimports=[
    'scipy.signal',
    'scipy.special',
    # ... other imports
]
```

#### 3. Audio Device Not Found

**Error**: "No audio devices available"

**Solution**:
- Ensure microphone/speakers are connected
- Check Windows audio settings
- Update audio drivers
- Run as administrator

#### 4. High CPU Usage

**Solution**:
- Reduce block size (increases latency): `block_size=1024`
- Use lower sample rate: `sample_rate=22050`
- Disable unused features in GUI

#### 5. Application Won't Start

**Solution**:
```bash
# Run with console to see errors
pyinstaller --name="AdvancedANCSystem" --console scripts/anc_gui_app.py

# Check error messages in console window
```

#### 6. Large File Size

**Solution**:
- Use UPX compression
- Exclude unnecessary libraries
- Use virtual environment to minimize dependencies

---

## Advanced Configuration

### Custom Icon

Add an icon to your executable:

1. Create or download a `.ico` file (256x256 recommended)
2. Update spec file:
   ```python
   exe = EXE(
       ...
       icon='path/to/icon.ico',
       ...
   )
   ```

### Version Information

Add version info to Windows executable:

1. Create `version.txt`:
   ```
   VSVersionInfo(
     ffi=FixedFileInfo(
       filevers=(1, 0, 0, 0),
       prodvers=(1, 0, 0, 0),
       ...
     ),
     ...
   )
   ```

2. Update spec file:
   ```python
   exe = EXE(
       ...
       version='version.txt',
       ...
   )
   ```

### Splash Screen

Add a splash screen while loading:

```python
splash = Splash(
    'path/to/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),
    text_size=12,
    text_color='white'
)
```

---

## Testing the Executable

### Pre-Distribution Checklist

- [ ] Test on clean Windows installation (no Python)
- [ ] Verify microphone access and audio playback
- [ ] Test all GUI features and modes
- [ ] Run for extended period (30+ minutes)
- [ ] Check CPU and memory usage
- [ ] Test on different Windows versions (10/11)
- [ ] Scan with antivirus (some may flag PyInstaller apps)

### Performance Benchmarks

Expected performance on modern hardware:
- **Startup time**: 2-5 seconds
- **Latency**: 40-50 ms
- **CPU usage**: 10-30% (single core)
- **Memory**: 100-200 MB
- **File size**: 200-400 MB

---

## Support and Updates

### Updating the Application

1. Make code changes
2. Increment version number
3. Rebuild with PyInstaller
4. Test thoroughly
5. Distribute new version

### Automatic Updates

For automatic update functionality, consider:
- **PyUpdater**: https://www.pyupdater.org/
- **Esky**: https://github.com/cloudmatrix/esky
- Custom update checker in your app

---

## License and Distribution

This software is part of the Active Noise Cancellation Capstone Project.

When distributing:
1. Include LICENSE file
2. Include README with usage instructions
3. Provide contact/support information
4. Consider digital signing for Windows (prevents security warnings)

---

## Additional Resources

- **PyInstaller Documentation**: https://pyinstaller.org/en/stable/
- **Python Packaging Guide**: https://packaging.python.org/
- **Inno Setup Documentation**: https://jrsoftware.org/ishelp/
- **Qt for Python**: https://doc.qt.io/qtforpython/

---

## Quick Reference

### Build Commands

```bash
# Standard build
pyinstaller anc_app_build.spec

# Clean build (removes cache)
pyinstaller --clean anc_app_build.spec

# Debug build (with console)
pyinstaller --debug all anc_app_build.spec
```

### File Locations

- **Executable**: `dist/AdvancedANCSystem/AdvancedANCSystem.exe`
- **Build logs**: `build/AdvancedANCSystem/`
- **Spec file**: `anc_app_build.spec`
- **Requirements**: `requirements_gui.txt`

---

**For questions or issues, please open an issue on GitHub or contact the development team.**
