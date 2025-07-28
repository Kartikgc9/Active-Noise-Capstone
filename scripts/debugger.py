"""
Diagnostic Script - Run this to identify the problem
"""

import sys
from pathlib import Path

print("🔍 DIAGNOSING AUDIO DENOISING SYSTEM")
print("=" * 45)

# Check Python version
print(f"Python version: {sys.version}")

# Check imports
modules_to_check = ['torch', 'librosa', 'soundfile', 'numpy', 'scipy']

for module in modules_to_check:
    try:
        __import__(module)
        print(f"✅ {module}: Available")
    except ImportError as e:
        print(f"❌ {module}: Missing - {e}")

# Check CUDA
try:
    import torch
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
except:
    print("❌ PyTorch not available")

# Check file structure
base_dir = Path.cwd()
important_paths = [
    base_dir / "audio_files" / "input",
    base_dir / "audio_files" / "output", 
    base_dir / "checkpoints"
]

for path in important_paths:
    if path.exists():
        print(f"✅ {path}: Exists")
    else:
        print(f"❌ {path}: Missing")

print("\n📋 Please share this output along with your error message!")
