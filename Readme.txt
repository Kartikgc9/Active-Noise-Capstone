1. Clone the repository:
bash
git clone https://github.com/Kartikgc9/Active-Noise-Capstone.git
cd Active-Noise-Capstone


2. Create and activate a virtual environment:
bash
python -m venv noise
.\noise\Scripts\activate  # Windows


3. Install required packages:
bash
pip install torch numpy librosa soundfile scipy


## Usage

### Basic Usage
Place your WAV files in the audio_files/input directory and run:
```bash
python scripts/audio_denoiser.py input_filename
