import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
import pandas as pd
from typing import Tuple, Optional
import random

class AudioDenoiseDataset(Dataset):
    """Dataset class for audio denoising training"""
    
    def __init__(self, 
                 data_path: Path,
                 metadata_file: Path,
                 sr: int = 44100,
                 segment_length: int = 44100,
                 train: bool = True,
                 augment: bool = True):
        """
        Initialize the dataset
        
        Args:
            data_path: Root path containing clean and noisy audio files
            metadata_file: Path to CSV file with clean/noisy audio pairs
            sr: Sample rate
            segment_length: Length of audio segments for training
            train: If True, enables random segmentation for training
            augment: If True, enables audio augmentation
        """
        self.data_path = Path(data_path)
        self.sr = sr
        self.segment_length = segment_length
        self.train = train
        self.augment = augment
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        print(f"📊 Loaded {len(self.metadata)} audio pairs")
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load and normalize audio file"""
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentation"""
        if not self.augment:
            return audio
            
        # Random gain
        gain = random.uniform(0.8, 1.2)
        audio = audio * gain
        
        # Random phase inversion
        if random.random() > 0.5:
            audio = -audio
            
        return audio
    
    def _get_segment(self, 
                    clean: np.ndarray, 
                    noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get random or sequential segment of audio"""
        if len(clean) < self.segment_length:
            # Pad if too short
            clean = np.pad(clean, (0, self.segment_length - len(clean)))
            noisy = np.pad(noisy, (0, self.segment_length - len(noisy)))
            return clean, noisy
            
        if self.train:
            # Random segment for training
            start = random.randint(0, len(clean) - self.segment_length)
            end = start + self.segment_length
        else:
            # Sequential segments for validation
            start = 0
            end = self.segment_length
            
        return clean[start:end], noisy[start:end]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training pair"""
        row = self.metadata.iloc[idx]
        
        # Load audio files
        clean_path = self.data_path / row["clean_file"]
        noisy_path = self.data_path / row["noisy_file"]
        
        clean = self._load_audio(clean_path)
        noisy = self._load_audio(noisy_path)
        
        # Get segments
        clean_seg, noisy_seg = self._get_segment(clean, noisy)
        
        # Augment if needed
        if self.augment and self.train:
            clean_seg = self._augment_audio(clean_seg)
            noisy_seg = self._augment_audio(noisy_seg)
        
        # Convert to tensors
        clean_tensor = torch.FloatTensor(clean_seg).unsqueeze(0)
        noisy_tensor = torch.FloatTensor(noisy_seg).unsqueeze(0)
        
        return noisy_tensor, clean_tensor  # Model input, target

def create_dataloaders(data_path: Path,
                      metadata_path: Path,
                      batch_size: int = 32,
                      sr: int = 44100,
                      segment_length: int = 44100,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        data_path: Root path containing audio files
        metadata_path: Path to metadata CSV
        batch_size: Batch size for training
        sr: Sample rate
        segment_length: Length of audio segments
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader: Training and validation dataloaders
    """
    # Training dataset
    train_dataset = AudioDenoiseDataset(
        data_path=data_path,
        metadata_file=metadata_path / "wav_train_pairs.csv",
        sr=sr,
        segment_length=segment_length,
        train=True,
        augment=True
    )
    
    # Validation dataset
    val_dataset = AudioDenoiseDataset(
        data_path=data_path,
        metadata_file=metadata_path / "wav_val_pairs.csv",
        sr=sr,
        segment_length=segment_length,
        train=False,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
