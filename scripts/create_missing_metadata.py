"""
Create Missing Metadata Directory and CSV Files
==============================================

This script creates the missing metadata directory and generates
CSV files from your existing spectrograms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Project paths
PROJECT_PATH = Path(__file__).parent.parent  # D:\Capstone
SPECTROGRAMS_PATH = PROJECT_PATH / 'spectrograms'
PROCESSED_PATH = PROJECT_PATH / 'processed'

def create_metadata_directory():
    """Create the missing processed/metadata directory"""
    metadata_dir = PROCESSED_PATH / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created metadata directory: {metadata_dir}")
    return metadata_dir

def find_matching_pairs():
    """Find matching clean/noisy spectrogram pairs"""
    print("🔍 Finding matching spectrogram pairs...")
    
    clean_dir = SPECTROGRAMS_PATH / 'clean'
    noisy_dir = SPECTROGRAMS_PATH / 'noisy'
    
    # Get all files
    clean_files = set([f.stem for f in clean_dir.glob('*.npy')])
    noisy_files = set([f.stem for f in noisy_dir.glob('*.npy')])
    
    print(f"📊 Found {len(clean_files)} clean spectrograms")
    print(f"📊 Found {len(noisy_files)} noisy spectrograms")
    
    # Find matching pairs
    matching_pairs = clean_files.intersection(noisy_files)
    print(f"🎯 Found {len(matching_pairs)} matching pairs")
    
    if len(matching_pairs) == 0:
        print("❌ No matching pairs found!")
        return None
    
    # Create pairs dataframe
    pairs_data = []
    for filename in sorted(matching_pairs):
        pairs_data.append({
            'clean_file': f'spectrograms/clean/{filename}.npy',
            'noisy_file': f'spectrograms/noisy/{filename}.npy'
        })
    
    pairs_df = pd.DataFrame(pairs_data)
    print(f"✅ Created {len(pairs_df)} pairs")
    
    return pairs_df

def split_dataset(pairs_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/validation/test sets"""
    from sklearn.model_selection import train_test_split
    
    print("📊 Splitting dataset...")
    
    # First split: separate training from (validation + test)
    train_df, temp_df = train_test_split(
        pairs_df, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Second split: separate validation from test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=42
    )
    
    print(f"  Training: {len(train_df)} pairs ({len(train_df)/len(pairs_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} pairs ({len(val_df)/len(pairs_df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} pairs ({len(test_df)/len(pairs_df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_csv_files(pairs_df, train_df, val_df, test_df, metadata_dir):
    """Save all CSV files to metadata directory"""
    print("💾 Saving CSV files...")
    
    # Save all pairs
    pairs_df.to_csv(metadata_dir / 'dataset_pairs.csv', index=False)
    print(f"  ✅ Saved: dataset_pairs.csv ({len(pairs_df)} pairs)")
    
    # Save splits
    train_df.to_csv(metadata_dir / 'train_pairs.csv', index=False)
    print(f"  ✅ Saved: train_pairs.csv ({len(train_df)} pairs)")
    
    val_df.to_csv(metadata_dir / 'val_pairs.csv', index=False)
    print(f"  ✅ Saved: val_pairs.csv ({len(val_df)} pairs)")
    
    test_df.to_csv(metadata_dir / 'test_pairs.csv', index=False)
    print(f"  ✅ Saved: test_pairs.csv ({len(test_df)} pairs)")

def main():
    """Main function to create missing metadata"""
    print("🚀 CREATING MISSING METADATA")
    print("=" * 40)
    
    # Step 1: Create metadata directory
    metadata_dir = create_metadata_directory()
    
    # Step 2: Find matching pairs from your spectrograms
    pairs_df = find_matching_pairs()
    
    if pairs_df is None or len(pairs_df) == 0:
        print("❌ Cannot proceed without matching pairs")
        return False
    
    # Step 3: Split dataset
    train_df, val_df, test_df = split_dataset(pairs_df)
    
    # Step 4: Save CSV files
    save_csv_files(pairs_df, train_df, val_df, test_df, metadata_dir)
    
    print(f"\n🎉 Metadata creation completed!")
    print(f"📁 Files created in: {metadata_dir}")
    print(f"🚀 You can now run the data loader again!")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Next step: Run data loader verification again:")
        print("   python scripts/demucs_dataloader_corrected.py")
    else:
        print("\n❌ Metadata creation failed. Check the errors above.")
