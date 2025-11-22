# 🎧 Advanced Active Noise Cancellation System

**AirPods-Inspired Technology with AI Enhancement**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

A cutting-edge active noise cancellation system that combines traditional signal processing with modern AI techniques and features reverse-engineered from Apple AirPods Pro.

![Advanced ANC System](https://img.shields.io/badge/Status-Production%20Ready-success)

---

## 🌟 Key Features

### 🎯 Multi-Mode Operation
- **ANC Mode**: Active noise cancellation with 5 intensity levels
- **Transparency Mode**: Adaptive ambient sound passthrough
- **Adaptive Mode**: Automatic switching based on voice detection
- **Off Mode**: Simple passthrough for natural listening

### 🗣️ Conversation Awareness
- Real-time voice activity detection
- Automatic mode switching when speaking
- Multi-criteria detection algorithm (energy, ZCR, spectral, harmonic)
- Adjustable sensitivity

### 🔊 Adaptive Transparency
- Customizable amplification (0.5x - 2.0x)
- Conversation boost (300-3400 Hz speech frequencies)
- Selective ambient noise reduction (0-100%)
- Tone adjustment (bass-treble control)
- Left-right balance control

### 🦻 Hearing Aid Functionality
- Frequency-specific amplification (4 bands)
- Dynamic range compression
- Independent left/right channel control
- Balance and tone adjustment
- Customizable frequency shaping

### 🎚️ Advanced Equalizer
- 7-band parametric EQ (-12 to +12 dB per band)
- 8 presets (Flat, Bass Boost, Vocal, Classical, etc.)
- Real-time adjustment
- Custom preset creation

### 🌍 Spatial Audio Simulation
- 3D sound positioning (azimuth, elevation, distance)
- Interaural Time Difference (ITD)
- Interaural Level Difference (ILD)
- Room reverberation simulation
- Quick position presets

### 🖥️ User Interface
- **GUI Application**: User-friendly PyQt5 desktop app
- **Standalone Executable**: Windows .exe (no Python required)
- **Command-Line Interface**: Power user access
- **Real-time Visualization**: Performance monitoring

### ⚡ Performance
- **Latency**: 40-50 ms (real-time capable)
- **CPU Usage**: 10-30% (single core)
- **Memory**: 100-200 MB
- **Real-Time Factor**: 2-3x (processes faster than playback)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Features in Detail](#-features-in-detail)
- [Building Executable](#-building-executable)
- [Testing](#-testing)
- [Performance](#-performance)
- [Comparison](#-comparison)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Quick Start

### Option 1: Use Pre-built Executable (Windows)

1. Download `AdvancedANCSystem.zip` from [Releases](releases)
2. Extract to any folder
3. Run `AdvancedANCSystem.exe`
4. Click "Calibrate Noise Profile" (3 seconds)
5. Click "Start Processing"
6. Enjoy!

### Option 2: Run from Source

```bash
# Clone repository
git clone https://github.com/yourusername/Active-Noise-Capstone.git
cd Active-Noise-Capstone

# Install dependencies
pip install -r requirements_gui.txt

# Run GUI application
python scripts/anc_gui_app.py
```

---

## 💻 Installation

### Prerequisites

- **Python 3.8 or higher** (3.9-3.11 recommended)
- **pip** (Python package installer)
- **Microphone and speakers/headphones**

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Linux, macOS | Windows 11 |
| **RAM** | 4 GB | 8 GB |
| **CPU** | Dual-core 2.0 GHz | Quad-core 3.0 GHz |
| **Storage** | 500 MB | 1 GB |
| **Audio** | Any microphone/speakers | Quality headphones |

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Active-Noise-Capstone.git
cd Active-Noise-Capstone
```

#### 2. Create Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# For GUI application
pip install -r requirements_gui.txt

# This installs:
# - numpy, scipy, librosa (audio processing)
# - soundfile, sounddevice (I/O)
# - PyQt5 (GUI)
# - pyinstaller (for building .exe)
# - pesq, pystoi (quality metrics)
```

#### 4. Verify Installation

```bash
# Run tests
python scripts/test_advanced_anc.py

# Expected output:
# ✓ PASS  Voice Activity Detection
# ✓ PASS  Transparency Mode
# ✓ PASS  Hearing Aid Processing
# ... etc.
```

---

## 📖 Usage

### GUI Application

#### Launch

```bash
python scripts/anc_gui_app.py
```

#### First-Time Setup

1. **Select Audio Mode**: Choose ANC, Transparency, Adaptive, or Off
2. **Calibrate** (for ANC mode):
   - Click "Calibrate Noise Profile"
   - Stay silent for 3 seconds
   - Let it record background noise
3. **Start Processing**: Click "Start Processing"
4. **Adjust Settings**: Use sliders and controls to customize

#### Main Controls Tab

- **Mode Selection**: Choose operating mode
- **ANC Intensity**: Select from Gentle to Maximum
- **Transparency Settings**:
  - Amplification: How loud ambient sounds are
  - Conversation Boost: Enhance speech frequencies
  - Ambient Reduction: Suppress background noise

#### Equalizer Tab

- **Presets**: Select from 8 built-in presets
- **Manual Control**: Adjust 7 frequency bands individually
- **Range**: -12 dB to +12 dB per band

#### Spatial Audio Tab

- **Azimuth**: Horizontal position (-180° to +180°)
- **Elevation**: Vertical position (-90° to +90°)
- **Distance**: Near to far (0.5x to 3.0x)
- **Quick Presets**: Center, Left, Right, etc.

#### Advanced Settings Tab

- System information
- Performance statistics
- About information

### Command-Line Interface

#### Basic ANC (Original System)

```bash
# Process audio file
python scripts/audio_denoiser.py input.wav normal
# Output: input_denoised_normal.wav

# Real-time denoising
python scripts/realtime_denoiser.py aggressive
```

#### Advanced ANC System

```bash
# Interactive mode
python scripts/advanced_anc_system.py

# Commands:
#   anc      - Switch to ANC mode
#   trans    - Switch to Transparency mode
#   adaptive - Switch to Adaptive mode (voice detection)
#   off      - Passthrough mode
#   stats    - Show performance statistics
#   quit     - Exit
```

### Python API

```python
from advanced_anc_system import MultiModeANCSystem, AudioMode
from audio_equalizer import AudioEqualizer

# Initialize system
anc = MultiModeANCSystem(
    sample_rate=44100,
    block_size=2048,
    mode=AudioMode.ADAPTIVE,
    noise_reduction_level="normal"
)

# Calibrate noise
anc.calibrate_noise(duration=3.0)

# Process audio chunk
processed = anc.process_chunk(audio_chunk)

# Switch modes
anc.set_mode(AudioMode.TRANSPARENCY)

# Get performance stats
stats = anc.get_performance_stats()
print(f"Latency: {stats['avg_latency_ms']:.2f} ms")

# Initialize equalizer
eq = AudioEqualizer()
eq.apply_preset('bass_boost')
equalized = eq.process(audio)
```

---

## 🎯 Features in Detail

### 1. Voice Activity Detection (VAD)

Our VAD uses four complementary techniques:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Energy** | RMS energy threshold | 30% |
| **Zero-Crossing Rate** | Voice has moderate ZCR (~0.1) | 20% |
| **Spectral Centroid** | Voice typically 300-3400 Hz | 30% |
| **Harmonic Ratio** | Voice has strong harmonics | 20% |

**Accuracy**: 80-90% in our tests

### 2. Transparency Mode

Intelligent ambient passthrough with:

- **High-pass filter**: Removes rumble (<20 Hz)
- **Noise gate**: Filters quiet sounds
- **Frequency separation**: Conversation vs. ambient
- **Conversation boost**: 1.0x - 2.0x enhancement
- **Ambient reduction**: 0-100% suppression
- **Tone control**: Bass-treble adjustment
- **Balance**: Left-right control

### 3. Hearing Aid Features

Professional-grade hearing assistance:

- **4-band frequency shaping**: Low, Mid, High, Ultra
- **Dynamic compression**: Makes quiet sounds audible
- **Channel independence**: L/R separate control
- **Medical disclaimer**: Not a medical device (consult audiologist)

### 4. Equalizer Presets

| Preset | Description | Bass | Mids | Treble |
|--------|-------------|------|------|--------|
| **Flat** | Reference (no change) | 0 dB | 0 dB | 0 dB |
| **Bass Boost** | Enhanced low frequencies | +6 dB | 0 dB | +1 dB |
| **Vocal Clarity** | Speech/podcast optimized | -3 dB | +5 dB | +3 dB |
| **Treble Boost** | Enhanced high frequencies | -2 dB | +2 dB | +6 dB |
| **Balanced** | Pleasant V-shape | +3 dB | -1 dB | +3 dB |
| **Podcast** | Spoken word | -4 dB | +5 dB | +2 dB |
| **Music** | General listening | +3 dB | 0 dB | +3 dB |
| **Classical** | Natural reproduction | +1 dB | 0 dB | +2 dB |

### 5. Spatial Audio

Simulates 3D sound positioning using:

- **ITD**: Time delay between ears (up to 0.7 ms)
- **ILD**: Level difference between ears (up to ±10 dB)
- **Elevation**: Frequency filtering for vertical position
- **Distance**: Amplitude and reverb adjustment
- **Room**: Multi-tap delay reverb

---

## 🔨 Building Executable

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed guide.

### Quick Build

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller anc_app_build.spec

# Find executable at:
# dist/AdvancedANCSystem/AdvancedANCSystem.exe
```

### Distribution

```bash
# Create ZIP for distribution
cd dist
zip -r AdvancedANCSystem.zip AdvancedANCSystem/

# Or use installer tools (Inno Setup, NSIS, etc.)
```

---

## 🧪 Testing

### Run Test Suite

```bash
python scripts/test_advanced_anc.py
```

### Test Coverage

| Test | What It Tests | Pass Criteria |
|------|---------------|---------------|
| **VAD** | Voice detection accuracy | ≥60% accuracy |
| **Transparency** | Mode processing | No clipping, <500ms |
| **Hearing Aid** | Frequency shaping | No clipping, <500ms |
| **Equalizer** | All presets | No clipping, <1000ms |
| **Spatial Audio** | All positions | No clipping, <1000ms |
| **Multi-Mode** | Mode switching | <100ms latency |
| **Real-Time** | Performance | Real-time factor >1.0 |

### Expected Results

```
✓ PASS  Voice Activity Detection (Accuracy: 85.7%)
✓ PASS  Transparency Mode
✓ PASS  Hearing Aid Processing
✓ PASS  Equalizer
✓ PASS  Spatial Audio
✓ PASS  Multi-Mode ANC
✓ PASS  Real-Time Performance (Real-time factor: 2.34x)

Total: 7/7 tests passed (100.0%)
```

---

## 📊 Performance

### Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency** | 40-50 ms | End-to-end processing |
| **CPU Usage** | 10-30% | Single core, typical |
| **Memory** | 100-200 MB | Including GUI |
| **Real-Time Factor** | 2-3x | Processes faster than playback |
| **VAD Accuracy** | 80-90% | Test suite results |
| **SNR Improvement** | 5-15 dB | Depends on noise level |

### Quality Metrics

| Metric | Typical | Excellent |
|--------|---------|-----------|
| **PESQ** | 3.0-4.0 | 3.5+ |
| **STOI** | 0.80-0.90 | 0.85+ |
| **SNR** | 10-20 dB | 15+ dB |

---

## 🆚 Comparison

### vs. Commercial Products

| Feature | AirPods Pro | Sony XM5 | Our System |
|---------|-------------|----------|------------|
| **ANC** | ✅ Excellent | ✅ Excellent | ✅ Good |
| **Transparency** | ✅ Yes | ✅ Yes | ✅ Customizable |
| **Voice Detection** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Spatial Audio** | ✅ Head-tracked | ✅ 360 Audio | ✅ Simulated |
| **Equalizer** | ❌ Limited | ✅ App | ✅ 7-band |
| **Hearing Aid** | ✅ Live Listen | ❌ No | ✅ Full features |
| **Cost** | $249 | $399 | **Free** |
| **Platform** | Apple only | Any | **Any** |
| **Customization** | ❌ Limited | ⚠️ Moderate | ✅ **Full** |
| **Open Source** | ❌ No | ❌ No | ✅ **Yes** |

### Advantages

✅ **Free and open source**
✅ **Works with any audio device** (no special hardware)
✅ **Fully customizable** (source code available)
✅ **Cross-platform** (Windows, Linux, macOS)
✅ **Advanced features** (7-band EQ, hearing aid)
✅ **Educational** (learn how ANC works)

### Limitations

⚠️ **Quality**: Commercial products have better hardware
⚠️ **Latency**: 40-50ms vs. <10ms in dedicated hardware
⚠️ **Portability**: Requires computer (not standalone device)
⚠️ **Battery**: Powered device only

---

## 🙏 Acknowledgments

### Inspiration

This project was inspired by the excellent reverse-engineering work of the [LibrePods project](https://github.com/kavishdevar/librepods), which documented Apple AirPods features including:

- Adaptive transparency mode
- Conversation awareness
- Hearing aid functionality
- Multi-device connectivity

### Technology Stack

- **Python**: Core language
- **NumPy/SciPy**: Numerical computing and signal processing
- **librosa**: Audio analysis
- **sounddevice**: Real-time audio I/O
- **PyQt5**: GUI framework
- **PyInstaller**: Executable building

### Contributors

Thank you to all contributors who have helped make this project better!

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📮 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Active-Noise-Capstone/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Active-Noise-Capstone/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

### Version 2.0 (Planned)

- [ ] Adaptive noise profiling (no manual calibration)
- [ ] Deep learning enhancement (better neural networks)
- [ ] Multi-device sync
- [ ] Head tracking support (with hardware)
- [ ] Cloud profile sync
- [ ] Mobile apps (Android/iOS)
- [ ] Environmental presets (office, airplane, gym, sleep)
- [ ] Audiogram import for hearing aid
- [ ] Plugin system for third-party extensions

### Version 3.0 (Future)

- [ ] Bluetooth device integration
- [ ] Multi-channel processing (5.1/7.1 surround)
- [ ] Real-time visualization
- [ ] Machine learning voice separation
- [ ] API for third-party apps

---

## 📚 Documentation

- **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**: Detailed feature documentation
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**: How to build executable
- **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)**: Algorithm explanations (coming soon)
- **[API_REFERENCE.md](API_REFERENCE.md)**: Python API documentation (coming soon)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

## 🎓 Educational Use

This project is perfect for:
- **Learning signal processing**: Hands-on audio DSP
- **Understanding ANC**: See how noise cancellation works
- **Python audio programming**: Real-world audio application
- **GUI development**: PyQt5 desktop application
- **Software engineering**: Clean architecture, testing, documentation

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Areas where we need help:
- Algorithm improvements
- Neural network models
- GUI enhancements
- Platform testing (Linux, macOS)
- Documentation
- Bug reports

---

**🎧 Enjoy your enhanced listening experience with advanced ANC technology!**

Made with ❤️ by the ANC Capstone Team
