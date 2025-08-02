#!/bin/bash
# Installation script for Whisper Voice-to-Text Application
# For Arch Linux with Hyprland

set -e

echo "Installing Whisper Voice-to-Text Application"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "bin/voice_transcriber.py" ]; then
    echo "Error: Please run this script from the whisper project directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install system dependencies for Arch Linux
echo "Installing system dependencies..."
echo "Please ensure you have the following packages installed:"
echo "- ffmpeg (for audio processing)"
echo "- libnotify (for desktop notifications)"
echo "- pulseaudio (for audio system)"
echo ""
echo "Install with: sudo pacman -S ffmpeg libnotify pulseaudio"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p config rec log

# Make scripts executable
chmod +x bin/voice_transcriber.py bin/whisper_daemon.py

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Ensure system audio dependencies are installed:"
echo "   sudo pacman -S ffmpeg libnotify pulseaudio portaudio"
echo ""
echo "2. Configure the application:"
echo "   source venv/bin/activate"
echo "   python bin/voice_transcriber.py --config"
echo ""
echo "3. Run the application:"
echo "   python bin/voice_transcriber.py"
echo ""
echo "4. For manual testing mode:"
echo "   python bin/voice_transcriber.py --manual"
echo ""
echo "Global hotkey: SUPER+A (may require additional Wayland configuration)"