# Active Noise Cancellation (ANC) System - Setup Guide

Complete setup instructions for running the ANC system on different platforms.

---

## 📋 **Prerequisites**

- **Python 3.8 or higher**
- **Audio input device** (microphone)
- **Audio output device** (speakers/headphones)
- **Internet connection** (for installing packages)

---

## 🚀 **Quick Setup (All Platforms)**

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone
```

### **Step 2: Install Dependencies**

Choose your platform and follow the instructions below.

---

## 💻 **Windows Setup**

### **Option 1: Automated Setup (Recommended)**

1. **Double-click** `setup_anc.bat` or run in CMD:
   ```cmd
   setup_anc.bat
   ```

2. **Follow the prompts**
   - Choose whether to create a virtual environment (recommended: Yes)
   - Wait for installation to complete

3. **Done!** Skip to "Running the System"

### **Option 2: Manual Setup**

```cmd
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install dependencies
pip install -r requirements_anc.txt

# 3. Verify installation
python -c "import sounddevice; print('Installation successful!')"
```

---

## 🐧 **Linux Setup**

### **Option 1: Automated Setup (Recommended)**

```bash
# Make script executable
chmod +x setup_anc.sh

# Run setup
./setup_anc.sh
```

### **Option 2: Manual Setup**

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev portaudio19-dev libsndfile1

# 2. Upgrade pip
pip3 install --upgrade pip

# 3. Install Python dependencies
pip3 install -r requirements_anc.txt

# 4. Verify installation
python3 -c "import sounddevice; print('Installation successful!')"
```

---

## 🍎 **macOS Setup**

### **Option 1: Automated Setup (Recommended)**

```bash
# Make script executable
chmod +x setup_anc.sh

# Run setup
./setup_anc.sh
```

### **Option 2: Manual Setup**

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install PortAudio
brew install portaudio

# 3. Upgrade pip
pip3 install --upgrade pip

# 4. Install Python dependencies
pip3 install -r requirements_anc.txt

# 5. Verify installation
python3 -c "import sounddevice; print('Installation successful!')"
```

---

## 🥧 **Raspberry Pi 5 Setup**

### **Important Notes for Raspberry Pi:**
- The ANC system works on Raspberry Pi 5
- Real-time processing may have higher latency due to CPU constraints
- USB audio devices recommended for better quality

### **Setup Instructions:**

```bash
# 1. Update system
sudo apt-get update
sudo apt-get upgrade

# 2. Install system dependencies
sudo apt-get install -y python3-pip python3-dev portaudio19-dev \
    libsndfile1 libffi-dev python3-numpy python3-scipy

# 3. Make script executable and run
chmod +x setup_anc.sh
./setup_anc.sh

# 4. (Optional) For better performance, increase audio buffer size
# Edit the script to use block_size=4096 instead of 2048
```

### **Performance Optimization for Raspberry Pi:**

If you experience choppy audio, edit `scripts/anc_system.py` line 495:

```python
# Change from:
block_size=2048,

# To:
block_size=4096,
```

This increases latency but reduces CPU usage.

---

## 📦 **What Gets Installed**

### **Core Dependencies (Required):**
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `librosa` - Audio analysis
- `soundfile` - Audio file I/O
- `sounddevice` - Real-time audio I/O

### **Optional Dependencies:**
- `tqdm` - Progress bars
- `audioread` - Additional audio format support

### **Total Download Size:** ~150-200 MB

---

## ✅ **Verify Installation**

After setup, test your installation:

### **1. Check Python Packages:**
```bash
python -c "import numpy, scipy, librosa, soundfile, sounddevice; print('✓ All packages installed')"
```

### **2. Check Audio Devices:**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

You should see a list of your audio input/output devices.

### **3. Run ANC System Tests:**
```bash
python scripts/test_anc_system.py
```

Expected output:
```
TEST SUMMARY
============================================================
✅ PASS: Initialization
✅ PASS: Denoising
✅ PASS: Delay Buffer
✅ PASS: All Levels

TOTAL: 4/4 tests passed (100.0%)
```

---

## 🎯 **Running the System**

### **Basic Usage:**
```bash
python scripts/anc_system.py gentle
```

### **All Options:**
```bash
python scripts/anc_system.py gentle      # Best performance (10.58 dB reduction)
python scripts/anc_system.py normal      # Balanced (9.60 dB reduction)
python scripts/anc_system.py moderate    # More aggressive
python scripts/anc_system.py aggressive  # Even more aggressive
python scripts/anc_system.py maximum     # Most aggressive
```

**Recommendation:** Start with `gentle` mode - it performs best based on testing!

---

## 🔧 **Troubleshooting**

### **Issue: "No module named 'sounddevice'"**

**Solution:**
```bash
pip install sounddevice
```

### **Issue: "PortAudio library not found" (Linux)**

**Solution:**
```bash
sudo apt-get install portaudio19-dev
pip install --force-reinstall sounddevice
```

### **Issue: "No audio devices found"**

**Solution:**
1. Check if microphone/speakers are connected
2. Check permissions:
   ```bash
   # Linux
   sudo usermod -a -G audio $USER
   # Then logout and login
   ```

### **Issue: "High CPU usage / choppy audio"**

**Solution:**
1. Use `gentle` mode (lowest CPU usage)
2. Increase block size in `anc_system.py`:
   ```python
   block_size=4096  # Instead of 2048
   ```
3. Close other applications

### **Issue: "Permission denied" on setup script**

**Solution:**
```bash
chmod +x setup_anc.sh
```

### **Issue: "pip: command not found"**

**Solution:**
```bash
# Try pip3 instead
pip3 install -r requirements_anc.txt
```

---

## 🌐 **VNC Viewer (Raspberry Pi)**

If you're using VNC Viewer to access Raspberry Pi:

1. **Audio won't work directly through VNC**
   - VNC doesn't forward audio
   - You need physical speakers/headphones connected to the Pi

2. **Options:**
   - Connect USB speakers/headphones to Raspberry Pi
   - Use HDMI audio output
   - Use 3.5mm audio jack

3. **Run the system on the Pi:**
   ```bash
   # SSH or VNC terminal
   python3 scripts/anc_system.py gentle
   ```

---

## 🔄 **Updating the Project**

When you pull latest changes from GitHub:

```bash
# Pull latest code
git pull origin test

# Update dependencies (if requirements changed)
pip install -r requirements_anc.txt

# Or re-run setup
./setup_anc.sh     # Linux/Mac/Raspberry Pi
setup_anc.bat      # Windows
```

---

## 📝 **Virtual Environment (Recommended)**

Using a virtual environment keeps dependencies isolated:

### **Create Virtual Environment:**

**Windows:**
```cmd
python -m venv anc_env
anc_env\Scripts\activate
```

**Linux/Mac/Raspberry Pi:**
```bash
python3 -m venv anc_env
source anc_env/bin/activate
```

### **Activate Later:**

**Windows:** `anc_env\Scripts\activate`
**Linux/Mac/Pi:** `source anc_env/bin/activate`

### **Deactivate:**
```bash
deactivate
```

---

## 📊 **System Requirements**

### **Minimum:**
- CPU: Dual-core 1.5 GHz
- RAM: 2 GB
- Python 3.8+
- Audio I/O devices

### **Recommended:**
- CPU: Quad-core 2.0 GHz or better
- RAM: 4 GB
- Python 3.9+
- USB audio interface (better quality)

### **Tested Platforms:**
- ✅ Windows 10/11
- ✅ Ubuntu 20.04/22.04
- ✅ macOS 11+
- ✅ Raspberry Pi 5 (with performance considerations)

---

## 🆘 **Getting Help**

If you encounter issues:

1. **Check this guide** - Most issues are covered above
2. **Run diagnostics:**
   ```bash
   python scripts/test_anc_system.py
   ```
3. **Check audio devices:**
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```
4. **Open an issue** on GitHub with:
   - Your platform (Windows/Linux/Mac/Raspberry Pi)
   - Python version (`python --version`)
   - Error message
   - Output of test script

---

## 📚 **Additional Resources**

- **Main README:** See `README.md` for project overview
- **Test Results:** See `TEST_RESULTS.md` for performance analysis
- **GitHub Repository:** https://github.com/Kartikgc9/Active-Noise-Capstone

---

## ⚡ **Quick Reference**

```bash
# Clone project
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone

# Setup (choose one)
./setup_anc.sh              # Linux/Mac/Raspberry Pi
setup_anc.bat               # Windows
pip install -r requirements_anc.txt  # Manual

# Run system
python scripts/anc_system.py gentle

# Test system
python scripts/test_anc_system.py

# Check devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

**Last Updated:** November 2025
**Version:** 1.0
**Tested On:** Windows 11, Ubuntu 22.04, Raspberry Pi 5
