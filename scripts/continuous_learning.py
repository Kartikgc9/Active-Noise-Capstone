"""
Continuous Learning System for Audio Denoiser
===========================================

This module implements automatic training based on new input/output pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import librosa
import soundfile as sf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from demucs_dataloader import SpectrogramPairDataset
from demucs_training_setup import AudioUNetWithDemucs
from audio_denoiser import AudioDenoiser

class AudioPairProcessor:
    """Processes new audio pairs and prepares them for training"""
    
    def __init__(self, input_dir, output_dir, spectrograms_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.spectrograms_dir = Path(spectrograms_dir)
        self.metadata_file = self.spectrograms_dir / "training_metadata.csv"
        self.processed_files = self._load_metadata()
    
    def _load_metadata(self):
        """Load or create training metadata"""
        if self.metadata_file.exists():
            return pd.read_csv(self.metadata_file)
        return pd.DataFrame(columns=["input_file", "output_file", "spectrogram_pair_id", 
                                   "quality_score", "processed_date"])
    
    def _save_metadata(self):
        """Save training metadata"""
        self.processed_files.to_csv(self.metadata_file, index=False)
    
    def process_audio_pair(self, input_file, output_file):
        """Process a new input/output pair for training"""
        # Generate unique ID for this pair
        pair_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Load audio files
            input_audio, sr = librosa.load(input_file, sr=44100)
            output_audio, _ = librosa.load(output_file, sr=44100)
            
            # Generate spectrograms
            input_spec = self._audio_to_spectrogram(input_audio)
            output_spec = self._audio_to_spectrogram(output_audio)
            
            # Save spectrograms
            np.save(self.spectrograms_dir / "noisy" / f"{pair_id}.npy", input_spec)
            np.save(self.spectrograms_dir / "clean" / f"{pair_id}.npy", output_spec)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(input_audio, output_audio)
            
            # Update metadata
            new_row = {
                "input_file": str(input_file),
                "output_file": str(output_file),
                "spectrogram_pair_id": pair_id,
                "quality_score": quality_score,
                "processed_date": datetime.now().isoformat()
            }
            self.processed_files = pd.concat([self.processed_files, 
                                            pd.DataFrame([new_row])],
                                           ignore_index=True)
            self._save_metadata()
            
            return True, pair_id
            
        except Exception as e:
            print(f"Error processing audio pair: {e}")
            return False, None
    
    def _audio_to_spectrogram(self, audio):
        """Convert audio to spectrogram"""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        spectrogram = np.abs(stft)
        return spectrogram
    
    def _calculate_quality_score(self, input_audio, output_audio):
        """Calculate a quality score for the denoising result"""
        # Calculate SNR improvement
        input_noise = input_audio - output_audio
        input_snr = 10 * np.log10(np.mean(output_audio**2) / np.mean(input_noise**2))
        return float(input_snr)

class ContinuousTrainer:
    """Manages continuous training of the denoising model"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.model = self._load_or_create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')
        
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        model = AudioUNetWithDemucs()
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model
    
    def train_on_new_data(self, dataset, batch_size=8, epochs=5):
        """Train model on new data"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (noisy, clean) in enumerate(dataloader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Save if better
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_checkpoint(avg_loss)
    
    def _save_checkpoint(self, loss):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'date': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.model_path)
        print(f"✅ Saved improved model with loss: {loss:.6f}")

class AudioFolderMonitor(FileSystemEventHandler):
    """Monitors audio folders for new files"""
    
    def __init__(self, processor, trainer):
        self.processor = processor
        self.trainer = trainer
        self.pending_pairs = {}
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.wav'):
            file_path = Path(event.src_path)
            if file_path.parent == self.processor.input_dir:
                # New input file
                self.pending_pairs[file_path.stem] = {'input': file_path}
            elif file_path.parent == self.processor.output_dir:
                # New output file
                base_name = file_path.stem
                if base_name.startswith('denoised_'):
                    base_name = base_name[9:]  # Remove 'denoised_' prefix
                if base_name in self.pending_pairs:
                    self.pending_pairs[base_name]['output'] = file_path
            
            # Check if we have a complete pair
            for base_name, paths in self.pending_pairs.items():
                if 'input' in paths and 'output' in paths:
                    success, pair_id = self.processor.process_audio_pair(
                        paths['input'], paths['output']
                    )
                    if success:
                        print(f"✅ Processed new audio pair: {base_name}")
                        # Create dataset and train
                        dataset = SpectrogramPairDataset(
                            self.processor.metadata_file,
                            self.processor.spectrograms_dir,
                            self.processor.processed_files,
                            augment=True
                        )
                        self.trainer.train_on_new_data(dataset)
                    del self.pending_pairs[base_name]

def start_continuous_learning():
    """Start the continuous learning system"""
    base_dir = Path(__file__).parent.parent
    audio_files_dir = base_dir / 'audio_files'
    input_dir = audio_files_dir / 'input'
    output_dir = audio_files_dir / 'output'
    spectrograms_dir = base_dir / 'spectrograms'
    model_path = base_dir / 'checkpoints' / 'test_model.pth'
    
    # Initialize components
    processor = AudioPairProcessor(input_dir, output_dir, spectrograms_dir)
    trainer = ContinuousTrainer(model_path)
    monitor = AudioFolderMonitor(processor, trainer)
    
    # Set up file system observer
    observer = Observer()
    observer.schedule(monitor, str(input_dir), recursive=False)
    observer.schedule(monitor, str(output_dir), recursive=False)
    
    print("🚀 Starting continuous learning system...")
    print(f"📂 Monitoring input directory: {input_dir}")
    print(f"📂 Monitoring output directory: {output_dir}")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n⏹️ Stopping continuous learning system...")
    observer.join()

if __name__ == "__main__":
    start_continuous_learning()
