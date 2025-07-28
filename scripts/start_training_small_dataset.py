"""
Fixed Training Script for Your Current Dataset
==============================================

Corrected import statements to match your actual file names.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path
from tqdm import tqdm

# Fixed imports to match your actual files
from demucs_training_setup import AudioUNetWithDemucs, EnhancedSpectrogramLoss
from demucs_dataloader import create_data_loaders_corrected  # Fixed import

def train_with_small_dataset():
    """Train with your current 38 pairs"""
    print("🚀 TRAINING WITH CURRENT DATASET (38 PAIRS)")
    print("=" * 50)
    
    # Create data loaders (exactly as tested)
    try:
        data_loaders, datasets = create_data_loaders_corrected(
            batch_size=2,      # Small batch size for limited data
            max_length=256,    # Matching your test configuration
            num_workers=0      # Single-threaded for stability
        )
        print("✅ Data loaders created successfully!")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    
    try:
        model = AudioUNetWithDemucs(
            n_channels=1,
            n_classes=1,
            use_demucs_features=True
        ).to(device)
        print("✅ Model initialized successfully!")
        
        # Print model info
        model_info = model.get_model_info()
        print(f"📊 Model parameters: {model_info['trainable_parameters']:,}")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # Training setup
    criterion = EnhancedSpectrogramLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    print(f"✅ Training setup complete!")
    print(f"📊 Training data: {len(datasets['train'])} samples")
    print(f"📊 Validation data: {len(datasets['val'])} samples")
    
    # Training loop (short test)
    num_epochs = 10  # Short training for testing
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Training
        try:
            for batch_idx, batch in enumerate(tqdm(data_loaders['train'], desc="Training")):
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)
                
                optimizer.zero_grad()
                predicted = model(noisy)
                loss_dict = criterion(predicted, clean)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loaders['train'])
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return
        
        # Validation
        model.eval()
        val_loss = 0.0
        try:
            with torch.no_grad():
                for batch in tqdm(data_loaders['val'], desc="Validation"):
                    noisy = batch['noisy'].to(device)
                    clean = batch['clean'].to(device)
                    predicted = model(noisy)
                    loss_dict = criterion(predicted, clean)
                    val_loss += loss_dict['total_loss'].item()
            
            avg_val_loss = val_loss / len(data_loaders['val'])
            scheduler.step(avg_val_loss)
            
            # Track best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"✅ New best validation loss: {best_val_loss:.4f}")
            
            print(f"📊 Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"📈 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return
    
    print(f"\n🎉 Training test completed successfully!")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Your complete pipeline is working perfectly!")
    
    # Save the trained model
    checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'epoch': num_epochs
    }, checkpoint_dir / 'test_model.pth')
    
    print(f"💾 Model saved to: {checkpoint_dir / 'test_model.pth'}")

if __name__ == "__main__":
    train_with_small_dataset()
