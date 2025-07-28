"""
U-Net Model for Audio Noise Reduction
====================================

This module implements a U-Net architecture specifically designed for 
audio spectrogram denoising using the VoiceBank + DEMAND dataset.

Architecture Features:
- Encoder-decoder structure with skip connections
- Optimized for spectrogram input (1025 frequency bins)
- Multiple loss functions for audio-specific optimization
- Frequency attention mechanism (optional)
- Memory-efficient design for Google Colab training

Author: Audio Noise Reduction Project
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    
    This is the fundamental building block of the U-Net architecture.
    Uses 3x3 convolutions with padding to maintain spatial dimensions.
    """
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool -> DoubleConv
    
    Reduces spatial dimensions by factor of 2 while increasing channel depth.
    Used in the encoder (contracting) path of U-Net.
    """
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample -> Concatenate -> DoubleConv
    
    Increases spatial dimensions by factor of 2 and combines with skip connection.
    Used in the decoder (expanding) path of U-Net.
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x1, x2):
        # Upsample x1
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2 (due to padding in convolutions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class FrequencyAttention(nn.Module):
    """
    Frequency-wise attention mechanism for audio spectrograms.
    
    Learns to focus on important frequency bands while suppressing noise.
    Specifically designed for audio processing where different frequencies
    have varying importance.
    """
    
    def __init__(self, channels, reduction_ratio=4):
        super(FrequencyAttention, self).__init__()
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # Pool across time dimension only
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.freq_attention(x)
        return x * attention_weights


class AudioUNet(nn.Module):
    """
    U-Net architecture optimized for audio spectrogram denoising.
    
    Input: Noisy magnitude spectrograms (batch_size, 1, freq_bins, time_frames)
    Output: Clean magnitude spectrograms (same shape as input)
    
    Architecture:
    - 5-level encoder with progressive downsampling
    - 4-level decoder with skip connections
    - Optional frequency attention mechanisms
    - Sigmoid output activation for normalized spectrograms
    
    Args:
        n_channels (int): Number of input channels (default: 1 for magnitude spectrograms)
        n_classes (int): Number of output channels (default: 1 for magnitude spectrograms)
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
        use_attention (bool): Enable frequency attention mechanisms
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, 
                 use_attention=False, dropout_rate=0.1):
        super(AudioUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64, dropout_rate)
        self.down1 = Down(64, 128, dropout_rate)
        self.down2 = Down(128, 256, dropout_rate)
        self.down3 = Down(256, 512, dropout_rate)
        
        # Adjust final down layer based on upsampling method
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate)
        
        # Frequency attention layers (optional)
        if use_attention:
            self.att1 = FrequencyAttention(128)
            self.att2 = FrequencyAttention(256)
            self.att3 = FrequencyAttention(512)
            self.att4 = FrequencyAttention(1024 // factor)
        
        # Decoder (Expanding Path)
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate)
        
        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # For normalized spectrograms [0,1]
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x (torch.Tensor): Input noisy spectrograms (batch_size, 1, freq_bins, time_frames)
            
        Returns:
            torch.Tensor: Predicted clean spectrograms (same shape as input)
        """
        
        # Encoder path with skip connections storage
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply frequency attention if enabled
        if self.use_attention:
            x2 = self.att1(x2)
            x3 = self.att2(x3)
            x4 = self.att3(x4)
            x5 = self.att4(x5)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)
        output = self.sigmoid(logits)
        
        return output
    
    def get_model_info(self):
        """Return model information including parameter count and memory usage."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'architecture': 'U-Net for Audio Denoising',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'uses_attention': self.use_attention,
            'upsampling_method': 'bilinear' if self.bilinear else 'transposed_conv'
        }


class SpectrogramLoss(nn.Module):
    """
    Combined loss function specifically designed for spectrogram-based audio denoising.
    
    Combines multiple loss components:
    1. MSE Loss: Overall magnitude reconstruction
    2. MAE Loss: Reduces over-smoothing, preserves sharp features
    3. Spectral Convergence: Ensures frequency domain fidelity
    4. Multi-scale STFT Loss: Perceptual audio quality (optional)
    
    Args:
        mse_weight (float): Weight for MSE loss component
        mae_weight (float): Weight for MAE loss component
        spectral_weight (float): Weight for spectral convergence loss
        stft_weight (float): Weight for multi-scale STFT loss
    """
    
    def __init__(self, mse_weight=1.0, mae_weight=0.1, spectral_weight=0.05, stft_weight=0.1):
        super(SpectrogramLoss, self).__init__()
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.spectral_weight = spectral_weight
        self.stft_weight = stft_weight
    
    def spectral_convergence_loss(self, predicted, target):
        """
        Spectral convergence loss for better frequency domain reconstruction.
        
        Args:
            predicted (torch.Tensor): Predicted spectrograms
            target (torch.Tensor): Target clean spectrograms
            
        Returns:
            torch.Tensor: Spectral convergence loss value
        """
        return torch.norm(target - predicted, p='fro') / (torch.norm(target, p='fro') + 1e-8)
    
    def multi_scale_stft_loss(self, predicted, target):
        """
        Multi-scale STFT loss for perceptual audio quality.
        Note: This is a simplified version. Full implementation would require
        converting spectrograms back to audio and computing STFT at multiple scales.
        """
        # Simplified version using different frequency band weights
        freq_bands = predicted.shape[2]
        low_freq = predicted[:, :, :freq_bands//4, :]
        mid_freq = predicted[:, :, freq_bands//4:3*freq_bands//4, :]
        high_freq = predicted[:, :, 3*freq_bands//4:, :]
        
        low_freq_target = target[:, :, :freq_bands//4, :]
        mid_freq_target = target[:, :, freq_bands//4:3*freq_bands//4, :]
        high_freq_target = target[:, :, 3*freq_bands//4:, :]
        
        low_loss = self.mae_loss(low_freq, low_freq_target)
        mid_loss = self.mae_loss(mid_freq, mid_freq_target)
        high_loss = self.mae_loss(high_freq, high_freq_target)
        
        # Weight different frequency bands
        return 0.5 * low_loss + 0.3 * mid_loss + 0.2 * high_loss
    
    def forward(self, predicted, target):
        """
        Compute combined loss.
        
        Args:
            predicted (torch.Tensor): Predicted clean spectrograms
            target (torch.Tensor): Target clean spectrograms
            
        Returns:
            dict: Dictionary containing all loss components
        """
        # Individual loss components
        mse = self.mse_loss(predicted, target)
        mae = self.mae_loss(predicted, target)
        spectral = self.spectral_convergence_loss(predicted, target)
        stft = self.multi_scale_stft_loss(predicted, target)
        
        # Combined total loss
        total_loss = (self.mse_weight * mse + 
                     self.mae_weight * mae + 
                     self.spectral_weight * spectral +
                     self.stft_weight * stft)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse,
            'mae_loss': mae,
            'spectral_loss': spectral,
            'stft_loss': stft
        }


def create_model(model_config=None):
    """
    Factory function to create AudioUNet model with specified configuration.
    
    Args:
        model_config (dict): Configuration dictionary for model parameters
        
    Returns:
        AudioUNet: Configured model instance
    """
    if model_config is None:
        model_config = {
            'n_channels': 1,
            'n_classes': 1,
            'bilinear': True,
            'use_attention': False,
            'dropout_rate': 0.1
        }
    
    model = AudioUNet(**model_config)
    return model


def test_model_architecture():
    """
    Test function to validate model architecture with dummy data.
    Useful for debugging and ensuring model works with expected input shapes.
    """
    print("Testing AudioUNet Architecture")
    print("=" * 40)
    
    # Create model
    model = create_model()
    model.eval()
    
    # Expected input shape based on VoiceBank + DEMAND analysis
    # n_fft=2048 -> freq_bins = 1025
    # ~4.35s audio at 48kHz with hop_length=512 -> ~410 time frames
    batch_size = 4
    freq_bins = 1025
    time_frames = 410
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 1, freq_bins, time_frames)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Shape consistency: {dummy_input.shape == output.shape}")
    
    # Model information
    model_info = model.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test loss function
    criterion = SpectrogramLoss()
    dummy_target = torch.rand_like(output)
    loss_dict = criterion(output, dummy_target)
    
    print(f"\nLoss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\n✅ Model architecture test completed successfully!")
    return model


if __name__ == "__main__":
    # Run architecture test when script is executed directly
    test_model = test_model_architecture()
