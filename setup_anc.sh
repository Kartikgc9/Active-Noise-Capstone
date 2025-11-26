#!/bin/bash
# Setup script for ANC System (Linux/Mac/Raspberry Pi)

echo "======================================================================="
echo "Active Noise Cancellation System - Setup"
echo "======================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Python 3 not found. Please install Python 3.8+"; exit 1; }
echo ""

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Detected: Raspberry Pi"
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-dev portaudio19-dev libsndfile1 libffi-dev
    echo ""
fi

# Create virtual environment (recommended)
echo "Do you want to create a virtual environment? (recommended) [y/n]"
read -r create_venv

if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv anc_env
    source anc_env/bin/activate
    echo "Virtual environment activated!"
    echo ""
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install minimal requirements for ANC system
echo "Installing ANC system requirements..."
pip install -r requirements_anc.txt
echo ""

# Test installation
echo "Testing installation..."
python3 -c "import numpy, scipy, librosa, soundfile, sounddevice; print('✓ All core libraries installed successfully!')" || {
    echo "✗ Installation test failed. Please check errors above."
    exit 1
}
echo ""

# Test audio devices
echo "Available audio devices:"
python3 -c "import sounddevice as sd; print(sd.query_devices())"
echo ""

echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo ""
echo "To run the ANC system:"
echo "  python3 scripts/anc_system.py gentle"
echo ""
echo "To test the system:"
echo "  python3 scripts/test_anc_system.py"
echo ""
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "Note: Remember to activate the virtual environment:"
    echo "  source anc_env/bin/activate"
    echo ""
fi
echo "======================================================================="
