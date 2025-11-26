#!/bin/bash
#
# Raspberry Pi 5 Setup Script for Noise Cancellation System
# ==========================================================
#
# This script installs all dependencies and configures
# the Raspberry Pi 5 for optimal noise cancellation performance.
#
# Usage: bash raspberry_pi_setup.sh
#

set -e  # Exit on error

echo "=========================================="
echo "🥧 Raspberry Pi 5 Noise Canceller Setup"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-scipy \
    portaudio19-dev \
    libsndfile1-dev \
    libasound2-dev \
    python3-venv \
    git

# Install audio tools
echo "📦 Installing audio tools..."
sudo apt-get install -y -qq \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install Python dependencies
echo "📦 Installing Python packages (this may take a while on Pi5)..."
echo "   Installing numpy..."
pip install numpy==1.24.3 -q

echo "   Installing scipy..."
pip install scipy==1.10.1 -q

echo "   Installing librosa..."
pip install librosa==0.10.0 -q

echo "   Installing soundfile..."
pip install soundfile==0.12.1 -q

echo "   Installing sounddevice..."
pip install sounddevice==0.4.6 -q

# Configure ALSA
echo "🔊 Configuring ALSA for low latency..."
if [ ! -f ~/.asoundrc ]; then
    cat > ~/.asoundrc << 'EOF'
# Low-latency ALSA configuration for noise cancellation
pcm.!default {
    type plug
    slave.pcm "hw:0,0"
}

ctl.!default {
    type hw
    card 0
}

# Low latency settings
defaults.pcm.rate_converter "speexrate_best"
EOF
    echo "   ✅ ALSA configured"
else
    echo "   ⚠️  ~/.asoundrc already exists, skipping"
fi

# Configure system for real-time audio
echo "⚡ Configuring system for real-time audio..."
sudo usermod -a -G audio $USER

# Set CPU governor to performance mode
echo "⚡ Setting CPU to performance mode..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo "   ✅ CPU governor set to performance"
else
    echo "   ⚠️  CPU governor not available"
fi

# Disable power management for USB devices (prevents audio dropouts)
echo "⚡ Configuring USB power management..."
sudo sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend' 2>/dev/null || true

# Create systemd service (optional)
echo "🚀 Creating systemd service..."
sudo tee /etc/systemd/system/noise-canceller.service > /dev/null << EOF
[Unit]
Description=Real-Time Noise Cancellation System
After=sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/scripts/pi5_noise_canceller.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "   ✅ Systemd service created"
echo "   To enable at boot: sudo systemctl enable noise-canceller"
echo "   To start now: sudo systemctl start noise-canceller"

# Test audio devices
echo ""
echo "🔊 Detecting audio devices..."
python3 -c "import sounddevice as sd; print(sd.query_devices())" 2>/dev/null || echo "⚠️  Could not query audio devices"

# Create startup script
echo "📝 Creating startup script..."
cat > start_noise_canceller.sh << 'EOF'
#!/bin/bash
# Quick start script for noise cancellation

cd "$(dirname "$0")"
source venv/bin/activate
python3 scripts/pi5_noise_canceller.py
EOF

chmod +x start_noise_canceller.sh

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo "   1. Connect your USB microphone"
echo "   2. Connect your USB speakers or use 3.5mm output"
echo "   3. Reboot for all changes to take effect: sudo reboot"
echo ""
echo "🚀 To start noise cancellation:"
echo "   ./start_noise_canceller.sh"
echo ""
echo "   Or directly:"
echo "   source venv/bin/activate"
echo "   python3 scripts/pi5_noise_canceller.py"
echo ""
echo "📖 For more options:"
echo "   python3 scripts/pi5_noise_canceller.py --help"
echo ""
echo "🔧 System service (auto-start at boot):"
echo "   sudo systemctl enable noise-canceller"
echo "   sudo systemctl start noise-canceller"
echo ""
