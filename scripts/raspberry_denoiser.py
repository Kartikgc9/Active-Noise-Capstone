"""
Raspberry Pi Audio Denoiser
=========================

Records audio from microphone and performs real-time denoising.
Requires:
- PyAudio for audio recording
- sounddevice for audio handling
- numpy for audio processing
"""

import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
from datetime import datetime
import librosa

from demucs_training_setup import AudioUNetWithDemucs

class RaspberryPiDenoiser:
    def __init__(self, model_path, sample_rate=44100, duration=30):
        """Initialize the denoiser with recording parameters"""
        self.sample_rate = sample_rate
        self.duration = duration
        self.device = torch.device('cpu')  # Raspberry Pi uses CPU
        
        # Load the model
        self.model_path = Path(model_path)
        self.model = self._load_model()
        print("✅ Model loaded successfully")
        
        # Recording parameters
        self.channels = 1  # Mono recording
        self.dtype = 'float32'
        
        # Setup directories
        self.base_dir = Path(__file__).parent.parent
        self.recordings_dir = self.base_dir / 'recordings'
        self.recordings_dir.mkdir(exist_ok=True)
        
    def _load_model(self):
        """Load the pre-trained model"""
        model = AudioUNetWithDemucs()
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        return model
    
    def record_audio(self):
        """Record audio from the microphone"""
        print(f"🎤 Recording for {self.duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()  # Wait until recording is done
        
        print("✅ Recording complete")
        return recording.flatten()
    
    def process_audio(self, audio):
        """Process the recorded audio through the model"""
        # Convert to spectrogram
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        spec = np.abs(stft).astype(np.float32)
        
        # Convert to tensor and add batch dimension
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        
        # Process through model
        with torch.no_grad():
            denoised_spec = self.model(spec_tensor)
        
        # Convert back to numpy
        denoised_spec = denoised_spec.squeeze(0).numpy()
        
        # Reconstruct audio
        denoised_audio = librosa.griffinlim(
            denoised_spec,
            n_iter=32,
            hop_length=256,
            win_length=1024
        )
        
        return denoised_audio
    
    def save_audio(self, audio, denoised_audio):
        """Save both original and denoised audio"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original
        original_path = self.recordings_dir / f"original_{timestamp}.wav"
        sf.write(original_path, audio, self.sample_rate)
        
        # Save denoised
        denoised_path = self.recordings_dir / f"denoised_{timestamp}.wav"
        sf.write(denoised_path, denoised_audio, self.sample_rate)
        
        return original_path, denoised_path
    
    def run_denoising_session(self):
        """Run a complete recording and denoising session"""
        try:
            # Record
            audio = self.record_audio()
            print("🔄 Processing audio...")
            
            # Denoise
            denoised_audio = self.process_audio(audio)
            
            # Save
            original_path, denoised_path = self.save_audio(audio, denoised_audio)
            print(f"✅ Saved recordings:")
            print(f"   Original: {original_path}")
            print(f"   Denoised: {denoised_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return False

def main():
    """Main function to run the Raspberry Pi denoiser"""
    # Initialize
    model_path = Path(__file__).parent.parent / 'checkpoints' / 'test_model.pth'
    denoiser = RaspberryPiDenoiser(model_path)
    
    print("🚀 Raspberry Pi Audio Denoiser")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            input("Press Enter to start recording...")
            denoiser.run_denoising_session()
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping denoiser...")

if __name__ == "__main__":
    main()
