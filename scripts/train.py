import torch
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import wandb

from data_processor import create_dataloaders
from models import AdvancedDenoiseNet, DenoisingLoss

class AudioDenoiseTrainer:
    """Trainer class for audio denoising model"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Create model
        self.model = AdvancedDenoiseNet(
            initial_channels=config['model']['initial_channels'],
            layers_per_block=config['model']['layers_per_block'],
            attention_layers=config['model']['attention_layers']
        ).to(self.device)
        
        # Setup loss and optimizer
        self.criterion = DenoisingLoss(
            alpha=config['training']['loss_alpha'],
            beta=config['training']['loss_beta'],
            window_size=config['training']['stft_window']
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Setup wandb for tracking
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                name=config['logging']['run_name'],
                config=config
            )
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (noisy, clean) in enumerate(pbar):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.config['logging']['use_wandb']:
                    wandb.log({'batch_loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for batch_idx, (noisy, clean) in enumerate(pbar):
                    noisy, clean = noisy.to(self.device), clean.to(self.device)
                    
                    output = self.model(noisy)
                    loss = self.criterion(output, clean)
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pth')
        
        # Save best model if needed
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f"✅ Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plots_dir = Path(self.config['paths']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'training_losses.png')
        plt.close()
    
    def train(self, resume_path: Optional[Path] = None):
        """Main training loop"""
        # Setup data
        train_loader, val_loader = create_dataloaders(
            data_path=Path(self.config['paths']['data_dir']),
            metadata_path=Path(self.config['paths']['metadata_dir']),
            batch_size=self.config['training']['batch_size'],
            sr=self.config['audio']['sample_rate'],
            segment_length=self.config['audio']['segment_length'],
            num_workers=self.config['training']['num_workers']
        )
        
        start_epoch = 0
        
        # Resume training if needed
        if resume_path and resume_path.exists():
            start_epoch, best_val_loss = self.load_checkpoint(resume_path)
            self.best_val_loss = best_val_loss
        
        print(f"🎯 Training for {self.config['training']['epochs']} epochs")
        
        try:
            for epoch in range(start_epoch, self.config['training']['epochs']):
                print(f"\n📊 Epoch {epoch+1}/{self.config['training']['epochs']}")
                
                # Training phase
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validation phase
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save checkpoints
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(epoch + 1, val_loss, is_best)
                
                # Log metrics
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Plot progress
                if (epoch + 1) % self.config['logging']['plot_interval'] == 0:
                    self.plot_losses()
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        
        finally:
            # Final cleanup
            if self.config['logging']['use_wandb']:
                wandb.finish()
            
            self.plot_losses()
            print("\n✅ Training completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")

def main():
    """Main training function"""
    # Load configuration
    config_path = Path(__file__).parent / "config" / "training_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Create trainer
    trainer = AudioDenoiseTrainer(config)
    
    # Start training
    resume_path = Path(config['paths']['checkpoint_dir']) / 'latest_model.pth'
    trainer.train(resume_path if resume_path.exists() else None)

if __name__ == "__main__":
    main()
