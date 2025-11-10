# Real-Time Audio Denoiser Setup Guide

This guide will help you set up and run the real-time audio denoising system.

## Prerequisites

- Python 3.8 or higher
- Working microphone
- Working speakers or headphones
- Stable system (Linux, Windows, or macOS)

## Installation Steps

### 1. Install Core Dependencies

```bash
# Option 1: Install all dependencies from requirements.txt
pip install -r requirements.txt

# Option 2: Install only what's needed for real-time processing
pip install numpy scipy librosa soundfile sounddevice
```

### 2. Install System Audio Libraries (if needed)

**On Linux:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install sounddevice
```

**On macOS:**
```bash
brew install portaudio
pip install sounddevice
```

**On Windows:**
```bash
# sounddevice should install directly
pip install sounddevice
```

### 3. Test Your Audio Setup

Run this command to list available audio devices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

You should see a list of your audio input (microphone) and output (speakers) devices.

## Running the Real-Time Denoiser

### Basic Usage

```bash
# Default (normal mode)
python scripts/realtime_denoiser.py

# With specific noise reduction level
python scripts/realtime_denoiser.py gentle
python scripts/realtime_denoiser.py moderate
python scripts/realtime_denoiser.py aggressive
python scripts/realtime_denoiser.py maximum
```

### What to Expect

1. **Startup**: The script will initialize and show your available audio devices
2. **Real-time Processing**: Audio will be continuously captured from your microphone
3. **Playback**: Denoised audio will be played back through your speakers/headphones
4. **Latency**: Expect ~46-92ms delay (similar to video call applications)
5. **Stopping**: Press `Ctrl+C` to stop the real-time denoising

### Tips for Best Results

1. **Use Headphones**: To prevent feedback loop between speakers and microphone
2. **Start with Normal Mode**: Begin with default settings and adjust as needed
3. **Monitor CPU Usage**: If you experience dropouts, the system may be overloaded
4. **Adjust Block Size**: For lower latency (edit `block_size` in the code)

## Troubleshooting

### Issue: "No module named 'sounddevice'"
**Solution**: Install sounddevice
```bash
pip install sounddevice
```

### Issue: "No module named 'librosa'"
**Solution**: Install audio processing libraries
```bash
pip install librosa soundfile scipy
```

### Issue: Audio Feedback/Echo
**Solution**: Use headphones instead of speakers

### Issue: High CPU Usage
**Solution**:
- Use a less aggressive noise reduction level (gentle or normal)
- Close other applications
- Increase block size for less frequent processing

### Issue: Choppy/Distorted Audio
**Solution**:
- Check that your system can handle real-time processing
- Try increasing the latency setting to 'high' (edit the code)
- Reduce other system load

### Issue: No Audio Output
**Solution**:
- Check that your speakers/headphones are properly connected
- Verify audio device selection in system settings
- Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to see devices

## How It Works

The real-time denoiser:

1. **Captures Audio**: Continuously records from microphone in small chunks (~46ms)
2. **Processing Thread**: Background thread processes each audio chunk
3. **Denoising**: Applies spectral subtraction and Wiener filtering
4. **Playback**: Immediately plays back the denoised audio

### Architecture

```
Microphone → Audio Callback → Input Queue → Processing Thread
                                                      ↓
                                            Denoise Audio Chunk
                                                      ↓
Speakers ← Audio Callback ← Output Queue ← Put Denoised Audio
```

### Performance

- **Latency**: ~46-92ms (buffer + processing time)
- **CPU Usage**: Moderate (one CPU core at 30-60%)
- **Memory**: ~100-200 MB
- **Sample Rate**: 44.1 kHz (CD quality)

## Advanced Configuration

To customize the real-time denoiser, edit `scripts/realtime_denoiser.py`:

```python
denoiser = RealtimeAudioDenoiser(
    noise_reduction_level="normal",  # gentle, normal, moderate, aggressive, maximum
    sample_rate=44100,               # Audio sample rate
    block_size=2048,                 # Smaller = lower latency, higher CPU
    latency='low'                    # 'low' or 'high'
)
```

### Block Size Trade-offs

- **1024 samples**: ~23ms latency, higher CPU usage
- **2048 samples**: ~46ms latency, moderate CPU usage (default)
- **4096 samples**: ~93ms latency, lower CPU usage

## Comparison with File-Based Processing

| Feature | Real-Time | File-Based |
|---------|-----------|------------|
| Latency | ~46-92ms | N/A |
| Use Case | Live audio | Post-processing |
| Quality | Good | Excellent |
| CPU Usage | Moderate | Low |
| Convenience | Automatic | Manual |

## Example Use Cases

1. **Video Calls**: Clean up your microphone audio in real-time
2. **Podcasting**: Monitor and clean audio during recording
3. **Live Streaming**: Reduce background noise during broadcasts
4. **Voice Recording**: Get immediate feedback on audio quality
5. **Noisy Environments**: Reduce ambient noise for clearer communication

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all dependencies are installed
3. Test with default settings first
4. Check system audio configuration

For additional help, please open an issue on GitHub.
