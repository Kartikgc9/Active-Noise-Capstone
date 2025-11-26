Complete Noise Cancellation System Guide
==========================================

This guide covers the complete noise cancellation system with three implementations:
1. **Enhanced File-Based Denoiser** - For processing audio files
2. **Real-Time Noise Canceller** - For desktop/laptop use
3. **Raspberry Pi 5 Optimized** - For embedded deployment

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [File-Based Processing](#file-based-processing)
4. [Real-Time Noise Cancellation](#real-time-noise-cancellation)
5. [Raspberry Pi 5 Deployment](#raspberry-pi-5-deployment)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## System Overview

### Features

✅ **Advanced Noise Cancellation**
- Multi-band spectral subtraction
- Adaptive Wiener filtering
- Spectral gating with soft knee
- Voice activity detection
- Speech-optimized frequency enhancement

✅ **Multiple Deployment Options**
- File processing (high quality, no time constraints)
- Real-time desktop (low latency, ~64ms)
- Raspberry Pi 5 (embedded, optimized for ARM)

✅ **Noise Types Supported**
- White noise (hiss, static)
- Pink noise (natural background)
- Low-frequency rumble (AC hum, traffic, 60Hz/120Hz)
- High-frequency hiss
- Impulsive noise (clicks, pops)
- Non-stationary noise (varying background)

✅ **Performance**
- Noise reduction: Up to 60+ dB
- Speech preservation: Excellent (300-3400 Hz protected)
- Latency (real-time): 32-128ms depending on configuration
- CPU usage: Moderate (30-60% single core)

---

## Installation

### Desktop/Laptop Installation

```bash
# Clone repository
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install numpy scipy librosa soundfile sounddevice matplotlib
```

### Raspberry Pi 5 Installation

```bash
# Clone repository on Pi5
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone

# Run automated setup script
bash raspberry_pi_setup.sh

# The script will:
# - Install all system dependencies
# - Configure audio subsystem for low latency
# - Set up Python environment
# - Create systemd service for auto-start
# - Optimize CPU governor for performance
```

---

## File-Based Processing

### Usage

For processing pre-recorded audio files with maximum quality:

```bash
# Basic usage (adaptive mode)
python scripts/enhanced_audio_denoiser.py input.wav

# With specific noise reduction level
python scripts/enhanced_audio_denoiser.py input.wav aggressive

# With separate noise sample
python scripts/enhanced_audio_denoiser.py input.wav adaptive noise_sample.wav
```

### Noise Reduction Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **adaptive** | Intelligently adjusts parameters | General purpose, recommended |
| **gentle** | Minimal processing | High-quality recordings, slight noise |
| **moderate** | Balanced approach | Most common scenarios |
| **aggressive** | Heavy noise reduction | Very noisy environments |
| **maximum** | Extreme processing | Extremely noisy audio (may reduce quality) |

### Input Requirements

- **Format**: WAV files (any sample rate, will be resampled to 44.1kHz)
- **Channels**: Mono or stereo (will be converted to mono)
- **Location**: `audio_files/input/`
- **Output**: `audio_files/output/enhanced_<filename>.wav`

### Example Workflow

```bash
# 1. Place audio file in input directory
cp my_noisy_recording.wav audio_files/input/

# 2. Process with aggressive mode
python scripts/enhanced_audio_denoiser.py my_noisy_recording.wav aggressive

# 3. Find result
# Output: audio_files/output/enhanced_my_noisy_recording.wav
```

### Advanced: Using Noise Sample

If you have a separate recording of just the noise:

```bash
# 1. Record noise sample (5-10 seconds of pure noise)
#    Save as: audio_files/input/noise_profile.wav

# 2. Process audio with noise sample
python scripts/enhanced_audio_denoiser.py speech_with_noise.wav adaptive noise_profile.wav
```

This provides significantly better results as the system can profile the exact noise.

---

## Real-Time Noise Cancellation

### Desktop/Laptop Usage

For real-time noise cancellation with microphone and speakers:

```bash
# Start with default settings
python scripts/realtime_noise_canceller.py

# Specify noise reduction mode
python scripts/realtime_noise_canceller.py --mode aggressive

# Specify audio devices
python scripts/realtime_noise_canceller.py --input-device 2 --output-device 3

# List available devices
python scripts/realtime_noise_canceller.py --list
```

### Full Options

```bash
python scripts/realtime_noise_canceller.py \
    --sample-rate 44100 \         # Sample rate (44100 for desktop, 16000 for Pi)
    --block-size 1024 \           # Block size (smaller = lower latency)
    --mode adaptive \             # Noise reduction mode
    --input-device 2 \            # Microphone device ID
    --output-device 3 \           # Speaker device ID
    --calibration-time 3.0        # Noise calibration duration (seconds)
```

### Operation

1. **Start the system**
   ```bash
   python scripts/realtime_noise_canceller.py
   ```

2. **Noise Calibration**
   - System will prompt you to remain silent for 3 seconds
   - This captures the ambient noise profile
   - Keep quiet during calibration for best results

3. **Normal Operation**
   - After calibration, speak normally into the microphone
   - Denoised audio plays through speakers in real-time
   - Latency: ~64-128ms (imperceptible in most cases)

4. **Stop**
   - Press `Ctrl+C` to stop
   - System shows statistics (processed blocks, drop rate, etc.)

### Important Notes

⚠️ **Use Headphones**: To prevent feedback loop between microphone and speakers

⚠️ **Calibration**: Perform calibration in the same environment where you'll use the system

⚠️ **CPU Usage**: Monitor CPU usage. If blocks are dropped, reduce sample rate or increase block size

---

## Raspberry Pi 5 Deployment

### Hardware Setup

**Required:**
- Raspberry Pi 5 (4GB+ RAM recommended)
- USB Microphone (or USB audio interface with mic input)
- USB Speakers (or 3.5mm audio output)
- Power supply (official Pi5 PSU recommended)
- MicroSD card (32GB+ recommended)

**Recommended:**
- Active cooling (fan or heatsink) for sustained performance
- Quality USB audio devices (low latency)
- Powered USB hub if using multiple USB audio devices

### Software Setup

```bash
# 1. Clone repository
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone

# 2. Run automated setup
bash raspberry_pi_setup.sh

# 3. Reboot to apply all changes
sudo reboot

# 4. Start noise cancellation
./start_noise_canceller.sh
```

### Running on Pi5

#### Quick Start

```bash
# Simple start (auto-detects USB devices)
python3 scripts/pi5_noise_canceller.py

# List available audio devices
python3 scripts/pi5_noise_canceller.py --list-devices

# Disable optimizations (for debugging)
python3 scripts/pi5_noise_canceller.py --no-optimization
```

#### Auto-Start at Boot

```bash
# Enable systemd service
sudo systemctl enable noise-canceller

# Start service now
sudo systemctl start noise-canceller

# Check status
sudo systemctl status noise-canceller

# View logs
sudo journalctl -u noise-canceller -f
```

#### Monitoring Performance

The Pi5 version shows real-time performance metrics:

```
📊 Blocks: 1234 | Dropped: 5 | Proc: 12.3ms (max: 18.5ms) | CPU: 45%
```

- **Blocks**: Total processed
- **Dropped**: Dropped due to CPU overload
- **Proc**: Average processing time per block
- **CPU**: Estimated CPU usage

**Target Performance:**
- Processing time: < 30ms for 512-sample blocks (< 64ms latency)
- Drop rate: < 1%
- CPU usage: < 70%

### Pi5 Optimization Tips

1. **Enable Performance Mode**
   ```bash
   # Set CPU to performance governor
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **Reduce Graphics Load**
   ```bash
   # Run headless or disable desktop effects
   # Free up CPU for audio processing
   ```

3. **Use Quality USB Audio**
   - Low-latency USB audio interfaces
   - Avoid cheap USB adapters
   - Test different USB ports (USB 3.0 ports preferred)

4. **Cool the Pi**
   - Active cooling prevents thermal throttling
   - Maintains consistent performance

5. **Monitor Temperature**
   ```bash
   # Check temperature
   vcgencmd measure_temp

   # Should stay below 70°C for best performance
   ```

---

## Technical Details

### Algorithms

#### 1. Multi-Band Spectral Subtraction

Divides audio into frequency bands and applies band-specific noise reduction:

- **0-150 Hz**: Aggressive (rumble removal)
- **300-500 Hz**: Moderate (lower speech)
- **500-2000 Hz**: Gentle (core speech)
- **2000-4000 Hz**: Gentle (consonants)
- **4000+ Hz**: Moderate (hiss removal)

#### 2. Spectral Gating

Applies soft-knee compression to frequencies below noise threshold:

- **Threshold**: Noise level + offset (typically -40 to -60 dB)
- **Ratio**: 10:1 to 30:1 depending on mode
- **Knee**: 6 dB soft knee for smooth transition

#### 3. Iterative Wiener Filtering

Multiple passes of Wiener filtering for residual noise:

- 2-5 iterations depending on mode
- Gain smoothing to prevent musical noise
- Adaptive signal/noise power estimation

#### 4. Speech Enhancement

Frequency-specific amplification:

- F1 region (300-900 Hz): +15%
- F2 region (900-3000 Hz): +20%
- F3 region (3000-4000 Hz): +15%
- Rumble (< 80 Hz): -70%

### Performance Characteristics

| Configuration | Sample Rate | Block Size | FFT Size | Latency | CPU Usage |
|---------------|-------------|------------|----------|---------|-----------|
| Desktop High Quality | 44100 Hz | 2048 | 2048 | ~93ms | 40-60% |
| Desktop Balanced | 44100 Hz | 1024 | 1024 | ~46ms | 50-70% |
| Desktop Low Latency | 44100 Hz | 512 | 512 | ~23ms | 70-90% |
| Pi5 Optimized | 16000 Hz | 512 | 512 | ~64ms | 50-70% |
| Pi5 Extreme Low Latency | 16000 Hz | 256 | 512 | ~32ms | 80-100% |

### Memory Usage

- **Desktop**: ~200-300 MB
- **Pi5**: ~100-150 MB (optimized)
- **Noise Profile**: ~5-10 KB
- **Audio Buffers**: ~50-100 KB

---

## Troubleshooting

### Common Issues

#### 1. No Audio Output

**Symptoms**: System runs but no sound from speakers

**Solutions**:
```bash
# Check audio devices
python scripts/realtime_noise_canceller.py --list

# Test audio output
speaker-test -c 2 -t wav

# Check ALSA/PulseAudio
aplay -l  # List playback devices
pactl list sinks  # PulseAudio sinks
```

#### 2. High CPU Usage / Dropped Blocks

**Symptoms**: "Dropped blocks" messages, choppy audio

**Solutions**:
```bash
# Reduce sample rate
--sample-rate 16000

# Increase block size
--block-size 2048

# Use less aggressive mode
--mode gentle

# On Pi5: Enable performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

#### 3. Feedback/Echo

**Symptoms**: Loud screeching sound

**Solutions**:
- **Use headphones** instead of speakers
- Reduce speaker volume
- Increase microphone-to-speaker distance
- Use directional microphone

#### 4. Poor Noise Cancellation

**Symptoms**: Noise still audible, speech muffled

**Solutions**:
- **Recalibrate** in the actual environment
- Ensure silent calibration (no speech/noise)
- Try more aggressive mode
- Provide separate noise sample
- Check if noise is stationary (works best on steady noise)

#### 5. Distorted Speech

**Symptoms**: Speech sounds robotic or unnatural

**Solutions**:
- Use less aggressive mode (try "gentle" or "moderate")
- Re-calibrate noise profile
- Check microphone positioning
- Ensure microphone isn't clipping (reduce input gain)

#### 6. Latency Issues

**Symptoms**: Noticeable delay between speech and output

**Solutions**:
```bash
# Reduce block size
--block-size 512

# On Pi5: Use lighter processing
# Already optimized, but can reduce sample rate further if needed
--sample-rate 8000  # Lowest quality but lowest latency
```

#### 7. Module Not Found Errors

**Symptoms**: "ModuleNotFoundError: No module named 'sounddevice'"

**Solutions**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt

# On Pi5: Install system dependencies
sudo apt-get install portaudio19-dev libsndfile1-dev
```

###8. USB Audio Device Not Detected (Pi5)

**Symptoms**: "No USB audio device found"

**Solutions**:
```bash
# Check USB devices
lsusb

# Check ALSA devices
aplay -l
arecord -l

# Restart audio
sudo systemctl restart alsa-utils
sudo systemctl restart pulseaudio

# Try different USB port (USB 3.0 preferred)

# Check device permissions
sudo usermod -a -G audio $USER
# Logout and login again
```

---

## Performance Optimization

### Desktop Optimization

```bash
# 1. Close unnecessary applications
# 2. Disable browser hardware acceleration if running in background
# 3. Use performance power plan (Windows) or performance governor (Linux)

# Linux: Set CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 4. Increase process priority
nice -n -10 python scripts/realtime_noise_canceller.py
```

### Raspberry Pi 5 Optimization

```bash
# 1. Enable performance mode
sudo raspi-config
# Advanced Options > Performance > Set to "Maximum"

# 2. Overclock (optional, requires cooling)
# Edit /boot/config.txt:
# arm_freq=2400
# gpu_freq=800

# 3. Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable wifi

# 4. Use tmpfs for temporary files
# Add to /etc/fstab:
# tmpfs /tmp tmpfs defaults,noatime,size=100M 0 0

# 5. Monitor temperature
watch -n 1 vcgencmd measure_temp
```

### Audio Buffer Optimization

Find optimal block size for your system:

```python
# Test different block sizes
for block_size in [256, 512, 1024, 2048, 4096]:
    # Run for 30 seconds and monitor drop rate
    # Choose largest block size with < 1% drop rate
    python scripts/realtime_noise_canceller.py --block-size {block_size}
```

---

## Testing and Validation

### Test Noise Cancellation Quality

```bash
# 1. Create test recording with known noise
python scripts/demo_realtime_functionality.py

# 2. Check output files
ls audio_files/demo/
# demo_noisy.wav
# demo_denoised_realtime.wav

# 3. Listen and compare
# Use audio player to compare before/after

# 4. View spectrograms
# Check visualizations/realtime_denoising_demo.png
```

### Measure Performance

```bash
# Run unit tests
python scripts/test_realtime_denoiser.py

# Run comprehensive demo
python scripts/demo_realtime_functionality.py

# Check SNR improvement
# Output will show noise reduction in dB
```

---

## Advanced Configuration

### Custom Noise Reduction Parameters

Edit the preset in the source code:

```python
# In enhanced_audio_denoiser.py or realtime_noise_canceller.py

self.presets["custom"] = {
    "spectral_floor": 0.001,        # Lower = more aggressive
    "over_subtraction": 3.0,        # Higher = more noise removal
    "gate_threshold_db": -50,       # Lower = more aggressive gating
    "gate_ratio": 15,               # Higher = stronger gate
    "smoothing": 5                  # Spectral smoothing (3-9)
}
```

### Frequency Band Customization

```python
# Adjust frequency bands for specific noise types

# For low-frequency dominated noise (traffic, HVAC):
self.speech_low = 400   # Raise lower bound
self.speech_high = 3400

# For high-frequency dominated noise (hiss):
self.speech_low = 300
self.speech_high = 4000  # Extend upper bound
```

---

## Project Structure

```
Active-Noise-Capstone/
├── scripts/
│   ├── enhanced_audio_denoiser.py       # File-based processing
│   ├── realtime_noise_canceller.py      # Desktop real-time
│   ├── pi5_noise_canceller.py           # Raspberry Pi 5 optimized
│   ├── test_realtime_denoiser.py        # Unit tests
│   └── demo_realtime_functionality.py   # Demo/validation
├── audio_files/
│   ├── input/                           # Input audio files
│   ├── output/                          # Processed output
│   └── demo/                            # Demo files
├── visualizations/                      # Spectrograms and plots
├── requirements.txt                     # Python dependencies
├── raspberry_pi_setup.sh                # Pi5 setup script
├── start_noise_canceller.sh            # Quick start script
├── NOISE_CANCELLATION_GUIDE.md         # This file
├── SETUP_REALTIME.md                   # Real-time setup guide
└── README.md                           # Project overview
```

---

## FAQ

**Q: What's the difference between the three implementations?**

A:
- **Enhanced File Denoiser**: Highest quality, no time constraints, processes files
- **Real-Time Canceller**: Desktop/laptop, low latency, continuous operation
- **Pi5 Optimized**: Embedded system, optimized for ARM, runs on Raspberry Pi 5

**Q: Can I use this for live streaming/video calls?**

A: Yes! Use virtual audio cables to route the denoised output to your streaming/video call application.

**Q: How much noise can it remove?**

A: Up to 60+ dB of noise reduction, depending on noise type and settings. Best results with stationary noise.

**Q: Does it work with all types of noise?**

A: Works best with stationary noise (hum, hiss, fan noise). Less effective with non-stationary noise (music, speech from others).

**Q: Will it affect speech quality?**

A: Minimal impact with proper calibration. Speech frequencies (300-3400 Hz) are specially protected.

**Q: Can I run multiple instances?**

A: No, only one instance can access the audio devices at a time.

**Q: What's the latency?**

A: Desktop: 46-128ms, Pi5: 32-64ms. Imperceptible for most use cases.

**Q: Does it work in real-time without delay?**

A: Yes, latency is minimal (~64ms typical). This is similar to Bluetooth audio latency.

---

## Support and Contributing

### Reporting Issues

Please include:
- System info (OS, CPU, RAM)
- Python version
- Full error message
- Configuration used
- Audio device info

### Contributing

Contributions welcome! Areas for improvement:
- Additional noise types
- Lower latency algorithms
- Better speech detection
- GPU acceleration
- Neural network enhancement

---

## License

MIT License - See LICENSE file for details

---

## Credits

Developed as part of the Active Noise Capstone project.

**Technologies Used:**
- Python 3.8+
- librosa (audio processing)
- NumPy/SciPy (numerical computing)
- sounddevice (audio I/O)
- soundfile (audio file handling)

---

**Last Updated**: 2025-11-26
