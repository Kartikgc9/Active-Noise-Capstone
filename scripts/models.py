import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConvBlock(nn.Module):
    """Basic convolutional block with residual connection"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection if needed
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + identity)

class AttentionBlock(nn.Module):
    """Self-attention block for temporal dependencies"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class AdvancedDenoiseNet(nn.Module):
    """Advanced audio denoising network with attention and skip connections"""
    def __init__(self, 
                 initial_channels: int = 32,
                 layers_per_block: int = 2,
                 attention_layers: bool = True):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Conv1d(1, initial_channels, 7, padding=3)
        
        # Encoder blocks with reduced channels
        self.enc1 = self._make_layer(initial_channels, initial_channels, layers_per_block)
        self.enc2 = self._make_layer(initial_channels, initial_channels, layers_per_block)
        self.enc3 = self._make_layer(initial_channels, initial_channels, layers_per_block)
        
        # Middle blocks with attention
        self.middle = nn.ModuleList([
            self._make_layer(initial_channels, initial_channels, layers_per_block),
            AttentionBlock(initial_channels) if attention_layers else nn.Identity()
        ])
        
        # Decoder blocks with matching channels
        self.dec3 = self._make_layer(initial_channels*2, initial_channels, layers_per_block)
        self.dec2 = self._make_layer(initial_channels*2, initial_channels, layers_per_block)
        self.dec1 = self._make_layer(initial_channels*2, initial_channels, layers_per_block)
        
        # Final convolution
        self.final_conv = nn.Conv1d(initial_channels, 1, 7, padding=3)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers = []
        layers.append(ConvBlock(in_channels, out_channels))
        
        for _ in range(1, blocks):
            layers.append(ConvBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x = self.init_conv(x)
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        e3 = self.enc3(F.avg_pool1d(e2, 2))
        
        # Middle blocks
        m = e3
        for layer in self.middle:
            m = layer(m)
            
        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(m, size=e2.shape[-1]), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[-1]), e1], dim=1))
        d1 = self.dec1(torch.cat([d2, x], dim=1))
        
        # Final convolution
        out = self.final_conv(d1)
        
        return out

class DenoisingLoss(nn.Module):
    """Combined loss function for audio denoising"""
    def __init__(self, 
                 alpha: float = 0.84,
                 beta: float = 0.16,
                 window_size: int = 2048):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        
        # L1 loss for time domain
        self.l1_loss = nn.L1Loss()
        
        # Initialize STFT for frequency domain loss
        self.register_buffer('window', torch.hann_window(window_size))
        
    def stft_loss(self, 
                  output: torch.Tensor, 
                  target: torch.Tensor) -> torch.Tensor:
        """Calculate loss in frequency domain"""
        output_stft = torch.stft(
            output.squeeze(1),
            n_fft=self.window_size,
            hop_length=self.window_size//4,
            window=self.window,
            return_complex=True
        )
        
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.window_size,
            hop_length=self.window_size//4,
            window=self.window,
            return_complex=True
        )
        
        mag_loss = F.l1_loss(output_stft.abs(), target_stft.abs())
        phase_loss = F.l1_loss(output_stft.angle(), target_stft.angle())
        
        return mag_loss + 0.1 * phase_loss
        
    def forward(self,
               output: torch.Tensor,
               target: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss"""
        # Time domain loss
        time_loss = self.l1_loss(output, target)
        
        # Frequency domain loss
        freq_loss = self.stft_loss(output, target)
        
        # Combine losses
        total_loss = self.alpha * time_loss + self.beta * freq_loss
        
        return total_loss
