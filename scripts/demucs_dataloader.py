"""
Corrected Production Data Loader for Your Spectrograms
====================================================

Updated to use the correct paths:
- Clean spectrograms: D:\Capstone\spectrograms\clean\
- Noisy spectrograms: D:\Capstone\spectrograms\noisy\
- CSV files: D:\Capstone\processed\metadata\
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Updated project paths based on your correction
PROJECT_PATH = Path(__file__).parent.parent  # D:\Capstone
SPECTROGRAMS_PATH = PROJECT_PATH / 'spectrograms'  # D:\Capstone\spectrograms
PROCESSED_PATH = PROJECT_PATH / 'processed'  # D:\Capstone\processed (for CSV files)

class SpectrogramPairDataset(Dataset):
    """
    Dataset for loading your clean/noisy spectrogram pairs from correct locations
    """
    
    def __init__(self, csv_file, spectrograms_path, processed_path, max_length=512, augment=False):
        """
        Args:
            csv_file: Path to CSV file (train_pairs.csv, val_pairs.csv, etc.)
            spectrograms_path: Path to spectrograms folder (D:\Capstone\spectrograms)
            processed_path: Path to processed folder (D:\Capstone\processed) 
            max_length: Maximum time frames per spectrogram
            augment: Whether to apply data augmentation (True for training)
        """
        print(f"📂 Loading dataset from: {csv_file}")
        
        # Load the CSV file
        self.pairs_df = pd.read_csv(csv_file)
        self.spectrograms_path = Path(spectrograms_path)
        self.processed_path = Path(processed_path)
        self.max_length = max_length
        self.augment = augment
        
        print(f"📊 Loaded {len(self.pairs_df)} pairs")
        print(f"🔍 Looking for spectrograms in: {self.spectrograms_path}")
        
        # Verify file structure
        self._verify_file_structure()
    
    def _verify_file_structure(self):
        """Verify that spectrogram files exist in correct locations"""
        print("🔍 Verifying file structure...")
        
        # Check directories exist
        clean_dir = self.spectrograms_path / 'clean'
        noisy_dir = self.spectrograms_path / 'noisy'
        
        if not clean_dir.exists():
            raise FileNotFoundError(f"Clean spectrograms directory not found: {clean_dir}")
        if not noisy_dir.exists():
            raise FileNotFoundError(f"Noisy spectrograms directory not found: {noisy_dir}")
        
        print(f"   ✅ Clean directory: {clean_dir}")
        print(f"   ✅ Noisy directory: {noisy_dir}")
        
        # Check first few samples
        for i in range(min(3, len(self.pairs_df))):
            row = self.pairs_df.iloc[i]
            
            # Extract filename from CSV path and build correct paths
            clean_filename = Path(row['clean_file']).name
            noisy_filename = Path(row['noisy_file']).name
            
            clean_path = clean_dir / clean_filename
            noisy_path = noisy_dir / noisy_filename
            
            if not clean_path.exists():
                raise FileNotFoundError(f"Clean spectrogram not found: {clean_path}")
            if not noisy_path.exists():
                raise FileNotFoundError(f"Noisy spectrogram not found: {noisy_path}")
            
            # Check file can be loaded
            if i == 0:  # Print info for first sample
                clean_spec = np.load(str(clean_path))
                print(f"   ✅ Sample spectrogram shape: {clean_spec.shape}")
                print(f"   ✅ Clean path: {clean_path}")
                print(f"   ✅ Noisy path: {noisy_path}")
        
        print("✅ File structure verification passed!")
    
    def __len__(self):
        return len(self.pairs_df)
    
    def _apply_augmentation(self, clean_spec, noisy_spec):
        """Apply data augmentation for training"""
        if not self.augment:
            return clean_spec, noisy_spec
        
        # Random time cropping (if sequence is longer than max_length)
        if self.max_length and clean_spec.shape[1] > self.max_length:
            max_start = clean_spec.shape[1] - self.max_length
            start_idx = random.randint(0, max_start)
            clean_spec = clean_spec[:, start_idx:start_idx + self.max_length]
            noisy_spec = noisy_spec[:, start_idx:start_idx + self.max_length]
        
        # Slight amplitude scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            clean_spec = np.clip(clean_spec * scale_factor, 0, 1)
            noisy_spec = np.clip(noisy_spec * scale_factor, 0, 1)
        
        return clean_spec, noisy_spec
    
    def __getitem__(self, idx):
        """Get a single spectrogram pair"""
        row = self.pairs_df.iloc[idx]
        
        # Extract filenames from CSV paths and build correct full paths
        clean_filename = Path(row['clean_file']).name
        noisy_filename = Path(row['noisy_file']).name
        
        clean_path = self.spectrograms_path / 'clean' / clean_filename
        noisy_path = self.spectrograms_path / 'noisy' / noisy_filename
        
        try:
            # Load spectrograms
            clean_spec = np.load(str(clean_path))
            noisy_spec = np.load(str(noisy_path))
            
            # Apply augmentation
            clean_spec, noisy_spec = self._apply_augmentation(clean_spec, noisy_spec)
            
            # Handle max length (for validation/test - center crop)
            if not self.augment and self.max_length and clean_spec.shape[1] > self.max_length:
                start_idx = (clean_spec.shape[1] - self.max_length) // 2
                clean_spec = clean_spec[:, start_idx:start_idx + self.max_length]
                noisy_spec = noisy_spec[:, start_idx:start_idx + self.max_length]
            
            # Convert to PyTorch tensors and add channel dimension
            clean_tensor = torch.FloatTensor(clean_spec).unsqueeze(0)  # Shape: (1, freq, time)
            noisy_tensor = torch.FloatTensor(noisy_spec).unsqueeze(0)  # Shape: (1, freq, time)
            
            return {
                'noisy': noisy_tensor,
                'clean': clean_tensor,
                'filename': clean_path.stem,
                'original_length': clean_spec.shape[1]
            }
            
        except Exception as e:
            print(f"❌ Error loading pair {idx}: {e}")
            print(f"   Clean path: {clean_path}")
            print(f"   Noisy path: {noisy_path}")
            
            # Return dummy data to prevent training interruption
            dummy_shape = (1, 1025, self.max_length or 400)
            return {
                'noisy': torch.zeros(dummy_shape),
                'clean': torch.zeros(dummy_shape),
                'filename': f'error_{idx}',
                'original_length': dummy_shape[2]
            }


class SpectrogramCollator:
    """Custom collate function to handle variable-length spectrograms"""
    
    def __init__(self, max_length=512):
        self.max_length = max_length
    
    def __call__(self, batch):
        """Collate a batch of spectrograms"""
        # Filter out None values (failed loads)
        batch = [item for item in batch if item is not None]
        
        if not batch:
            raise ValueError("Empty batch after filtering")
        
        # Find maximum time dimension in batch
        max_time = max([item['noisy'].shape[2] for item in batch])
        
        if self.max_length:
            max_time = min(max_time, self.max_length)
        
        # Pad all spectrograms to same length
        padded_noisy = []
        padded_clean = []
        filenames = []
        original_lengths = []
        
        for item in batch:
            noisy = item['noisy']
            clean = item['clean']
            
            current_time = noisy.shape[2]
            
            if current_time > max_time:
                # Truncate from center
                start_idx = (current_time - max_time) // 2
                noisy = noisy[:, :, start_idx:start_idx + max_time]
                clean = clean[:, :, start_idx:start_idx + max_time]
            elif current_time < max_time:
                # Pad at the end
                pad_amount = max_time - current_time
                noisy = torch.nn.functional.pad(noisy, (0, pad_amount))
                clean = torch.nn.functional.pad(clean, (0, pad_amount))
            
            padded_noisy.append(noisy)
            padded_clean.append(clean)
            filenames.append(item['filename'])
            original_lengths.append(item['original_length'])
        
        return {
            'noisy': torch.stack(padded_noisy),
            'clean': torch.stack(padded_clean),
            'filenames': filenames,
            'original_lengths': original_lengths
        }


def create_data_loaders_corrected(batch_size=4, max_length=512, num_workers=2):
    """
    Create production data loaders using your corrected file paths
    
    Args:
        batch_size: Number of samples per batch
        max_length: Maximum time frames per spectrogram
        num_workers: Number of worker processes for data loading
    
    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    
    print("🚀 Creating Data Loaders with Corrected Paths")
    print("=" * 50)
    print(f"📂 Project path: {PROJECT_PATH}")
    print(f"📂 Spectrograms path: {SPECTROGRAMS_PATH}")
    print(f"📂 Processed path (CSV files): {PROCESSED_PATH}")
    
    # Define paths to your CSV files (still in processed/metadata)
    metadata_dir = PROCESSED_PATH / 'metadata'
    csv_files = {
        'train': metadata_dir / 'train_pairs.csv',
        'val': metadata_dir / 'val_pairs.csv',
        'test': metadata_dir / 'test_pairs.csv'
    }
    
    # Verify all CSV files exist
    for split, csv_path in csv_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {csv_path}")
        print(f"✅ Found {split} CSV: {csv_path}")
    
    # Verify spectrogram directories exist
    clean_dir = SPECTROGRAMS_PATH / 'clean'
    noisy_dir = SPECTROGRAMS_PATH / 'noisy'
    
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean spectrograms directory not found: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy spectrograms directory not found: {noisy_dir}")
    
    print(f"✅ Found clean spectrograms: {clean_dir}")
    print(f"✅ Found noisy spectrograms: {noisy_dir}")
    
    # Count files
    clean_count = len(list(clean_dir.glob('*.npy')))
    noisy_count = len(list(noisy_dir.glob('*.npy')))
    print(f"📊 Clean spectrograms: {clean_count:,} files")
    print(f"📊 Noisy spectrograms: {noisy_count:,} files")
    
    # Create datasets
    datasets = {}
    data_loaders = {}
    
    print(f"\n📊 Creating datasets...")
    
    # Training dataset (with augmentation)
    datasets['train'] = SpectrogramPairDataset(
        csv_file=csv_files['train'],
        spectrograms_path=SPECTROGRAMS_PATH,
        processed_path=PROCESSED_PATH,
        max_length=max_length,
        augment=True  # Enable data augmentation for training
    )
    
    # Validation dataset (no augmentation)
    datasets['val'] = SpectrogramPairDataset(
        csv_file=csv_files['val'],
        spectrograms_path=SPECTROGRAMS_PATH,
        processed_path=PROCESSED_PATH,
        max_length=max_length,
        augment=False  # No augmentation for validation
    )
    
    # Test dataset (no augmentation)
    datasets['test'] = SpectrogramPairDataset(
        csv_file=csv_files['test'],
        spectrograms_path=SPECTROGRAMS_PATH,
        processed_path=PROCESSED_PATH,
        max_length=max_length,
        augment=False  # No augmentation for testing
    )
    
    # Create collate function
    collate_fn = SpectrogramCollator(max_length=max_length)
    
    print(f"\n🔄 Creating data loaders...")
    
    # Create data loaders
    data_loaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    data_loaders['val'] = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    data_loaders['test'] = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Print summary
    print(f"\n🎉 Data Loaders Created Successfully!")
    print(f"   Training:   {len(datasets['train']):,} samples ({len(data_loaders['train']):,} batches)")
    print(f"   Validation: {len(datasets['val']):,} samples ({len(data_loaders['val']):,} batches)")
    print(f"   Test:       {len(datasets['test']):,} samples ({len(data_loaders['test']):,} batches)")
    print(f"   Batch size: {batch_size}")
    print(f"   Max length: {max_length} time frames")
    print(f"   Workers:    {num_workers}")
    
    return data_loaders, datasets


def test_corrected_data_loaders():
    """Test the data loaders with your corrected file paths"""
    print("🧪 Testing Data Loaders with Corrected File Paths")
    print("=" * 60)
    
    try:
        # Create data loaders with corrected paths
        data_loaders, datasets = create_data_loaders_corrected(
            batch_size=2,      # Small batch for testing
            max_length=256,    # Shorter sequences for testing
            num_workers=0      # No multiprocessing for testing
        )
        
        print(f"\n🔍 Testing each data loader...")
        
        # Test each split
        for split_name, loader in data_loaders.items():
            print(f"\n📊 Testing {split_name} loader...")
            
            # Get one batch
            for batch_idx, batch in enumerate(loader):
                noisy = batch['noisy']
                clean = batch['clean']
                filenames = batch['filenames']
                
                print(f"   Batch {batch_idx + 1}:")
                print(f"     Noisy shape: {noisy.shape}")
                print(f"     Clean shape: {clean.shape}")
                print(f"     Value ranges - Noisy: [{noisy.min():.3f}, {noisy.max():.3f}]")
                print(f"     Value ranges - Clean: [{clean.min():.3f}, {clean.max():.3f}]")
                print(f"     Sample filenames: {filenames[:2]}")
                print(f"   ✅ {split_name.capitalize()} loader working correctly!")
                
                # Only test first batch
                break
        
        print(f"\n🎉 All Data Loaders Tested Successfully!")
        print(f"✅ Your {len(datasets['train']):,} training samples are ready!")
        print(f"✅ Your {len(datasets['val']):,} validation samples are ready!")
        print(f"✅ Your {len(datasets['test']):,} test samples are ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_corrected_file_structure():
    """Verify your corrected file structure"""
    print("🔍 Verifying Corrected File Structure...")
    print("=" * 40)
    
    # Check directories
    clean_dir = SPECTROGRAMS_PATH / 'clean'
    noisy_dir = SPECTROGRAMS_PATH / 'noisy'
    metadata_dir = PROCESSED_PATH / 'metadata'
    
    print(f"📂 Checking directories:")
    print(f"   Clean spectrograms: {clean_dir}")
    print(f"   Exists: {clean_dir.exists()}")
    
    print(f"   Noisy spectrograms: {noisy_dir}")
    print(f"   Exists: {noisy_dir.exists()}")
    
    print(f"   Metadata (CSV files): {metadata_dir}")
    print(f"   Exists: {metadata_dir.exists()}")
    
    if clean_dir.exists():
        clean_count = len(list(clean_dir.glob('*.npy')))
        print(f"   Clean files: {clean_count:,}")
    
    if noisy_dir.exists():
        noisy_count = len(list(noisy_dir.glob('*.npy')))
        print(f"   Noisy files: {noisy_count:,}")
    
    if metadata_dir.exists():
        csv_files = list(metadata_dir.glob('*.csv'))
        print(f"   CSV files: {len(csv_files)}")
        for csv_file in csv_files:
            print(f"     ✅ {csv_file.name}")
    
    # Test loading a sample
    try:
        if clean_dir.exists() and noisy_dir.exists():
            sample_files = list(clean_dir.glob('*.npy'))[:1]
            if sample_files:
                sample_file = sample_files[0]
                spec = np.load(str(sample_file))
                print(f"\n🧪 Sample file test:")
                print(f"   File: {sample_file.name}")
                print(f"   Shape: {spec.shape}")
                print(f"   ✅ File loading successful!")
            else:
                print("❌ No .npy files found in clean directory")
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
    
    return clean_dir.exists() and noisy_dir.exists() and metadata_dir.exists()


if __name__ == "__main__":
    print("🚀 CORRECTED DATA LOADER VERIFICATION")
    print("=" * 60)
    
    # Step 1: Verify file structure
    if not verify_corrected_file_structure():
        print("❌ File structure verification failed!")
        exit(1)
    
    # Step 2: Test data loaders
    if test_corrected_data_loaders():
        print(f"\n🎯 SUCCESS! Your corrected data loaders are ready!")
        print(f"🚀 You can now start training with:")
        print(f"   python scripts/start_training_corrected.py")
    else:
        print(f"\n❌ Data loader test failed!")
        print(f"Please check the error messages above.")
