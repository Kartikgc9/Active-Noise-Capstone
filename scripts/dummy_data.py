import torch

# Example parameters matching your pipeline
batch_size = 4
freq_bins = 1025        # n_fft//2 + 1 if n_fft=2048
time_steps = 400        # try 300-500, adjust as needed

# Create random input tensor simulating your normalized spectrograms
dummy_input = torch.randn(batch_size, 1, freq_bins, time_steps)

# Import and build your U-Net model class (from unet_model.py)
from unet_model import AudioUNet


# Instantiate model
model = AudioUNet(n_channels=1, n_classes=1, bilinear=True)

# Put model in evaluation mode and test forward pass
model.eval()
with torch.no_grad():
    dummy_output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", dummy_output.shape)
assert dummy_input.shape == dummy_output.shape, "Input/output shapes do not match!"
