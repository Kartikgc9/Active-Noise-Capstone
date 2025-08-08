# Active Noise Reduction System
An advanced audio denoising system that combines traditional signal processing techniques with neural network enhancement capabilities.

## Project Overview
This system provides multiple levels of noise reduction for audio files, ranging from gentle to maximum noise reduction. It uses a combination of:
- Spectral Subtraction
- Wiener Filtering
- Adaptive Noise Reduction
- Optional Neural Network Enhancement

## Directory Structure
```
audio_files/
    input/      # Place input WAV files here
    output/     # Processed files will be saved here
checkpoints/    # Neural network model checkpoints
metadata/       # Dataset metadata
scripts/        # Core processing scripts
spectrograms/   # Spectrogram data for training
```

## Requirements
- Python 3.8 or higher
- PyTorch
- librosa
- soundfile
- numpy
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone
```

2. Create and activate a virtual environment:
```bash
python -m venv noise
.\noise\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install torch numpy librosa soundfile scipy
```

## Usage

### Basic Usage
Place your WAV files in the `audio_files/input` directory and run:
```bash
python scripts/audio_denoiser.py input_filename
```

### Noise Reduction Levels
The system supports five noise reduction levels:
- `gentle`: Minimal noise reduction, preserves audio quality
- `normal`: Balanced noise reduction (default)
- `moderate`: Stronger noise reduction
- `aggressive`: Heavy noise reduction
- `maximum`: Maximum possible noise reduction

To specify a noise reduction level:
```bash
python scripts/audio_denoiser.py input_filename reduction_level
```

Examples:
```bash
python scripts/audio_denoiser.py input3.wav               # Normal mode
python scripts/audio_denoiser.py input3.wav gentle        # Gentle mode
python scripts/audio_denoiser.py input3.wav moderate      # Moderate mode
python scripts/audio_denoiser.py input3.wav aggressive    # Aggressive mode
python scripts/audio_denoiser.py input3.wav maximum       # Maximum mode
```

### Output
- Processed files are saved in `audio_files/output` directory
- Output filenames are prefixed with the reduction level used (e.g., `normal_input3.wav`)
- The system provides detailed processing statistics and quality metrics during execution

## Features

### 1. Adaptive Processing
- Automatically analyzes audio characteristics
- Adjusts processing based on audio content
- Provides real-time feedback on processing decisions

### 2. Multi-stage Processing
- Advanced spectral subtraction
- Enhanced Wiener filtering
- Adaptive noise reduction
- Neural network enhancement (when model available)

### 3. Quality Preservation
- Maintains original audio length
- Preserves speech intelligibility
- Prevents over-processing artifacts
- Dynamic parameter adjustment

### 4. Performance Metrics
- RMS energy monitoring
- Signal-to-noise ratio estimation
- Processing quality assessment
- Detailed execution logging

## Performance Considerations

### Audio Characteristics
- Best results with clear speech input
- Effective on steady-state background noise
- Adaptive to varying noise levels
- Preserves speech frequencies (300-3400 Hz)

### Processing Levels
1. **Gentle**
   - Minimal intervention
   - Best for high-quality recordings
   - Preserves subtle audio details

2. **Normal**
   - Balanced processing
   - Suitable for most recordings
   - Good noise reduction without artifacts

3. **Moderate**
   - Enhanced noise reduction
   - More aggressive processing
   - May introduce slight artifacts

4. **Aggressive**
   - Heavy noise reduction
   - Multi-stage processing
   - Best for very noisy recordings

5. **Maximum**
   - Strongest possible processing
   - Use only for severe noise cases
   - May affect audio quality

## Troubleshooting

### Common Issues
1. **Low Output Volume**
   - Check input audio normalization
   - Verify RMS levels in processing output
   - Consider using a less aggressive mode

2. **Processing Artifacts**
   - Try a gentler noise reduction level
   - Check input audio quality
   - Ensure proper input format (WAV)

3. **Error Messages**
   - Verify Python environment setup
   - Check input file format and existence
   - Ensure sufficient system resources

### Best Practices
1. **Input Audio**
   - Use WAV format files
   - Ensure proper sample rate (44.1kHz)
   - Normalize audio if necessary

2. **Processing**
   - Start with 'normal' mode
   - Monitor processing statistics
   - Adjust level based on results

3. **Output Quality**
   - Compare with original audio
   - Check for artifacts
   - Verify speech clarity

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
