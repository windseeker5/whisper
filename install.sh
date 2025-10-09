#!/bin/bash
# Installation script for Voice-to-Text Application
# For Arch Linux with Sway/Hyprland

set -e

echo "=================================================="
echo "  Voice-to-Text Application Installation"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "bin/voice_transcriber.py" ]; then
    echo "❌ Error: Please run this script from the whisper project directory"
    exit 1
fi

# Check system dependencies
echo "Step 1: Checking system dependencies..."
echo ""
MISSING_DEPS=""

for pkg in ffmpeg libnotify portaudio; do
    if ! pacman -Q "$pkg" &>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "❌ Missing system dependencies:$MISSING_DEPS"
    echo ""
    echo "Install with:"
    echo "  sudo pacman -S$MISSING_DEPS"
    echo ""
    exit 1
else
    echo "✓ All system dependencies found"
fi

# Check if virtual environment exists
echo ""
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies (WITHOUT whisper/torch)
echo ""
echo "Step 3: Installing core dependencies + Vosk backend..."
echo "(This is lightweight - NO torch download, NO timeout issues)"
echo ""
pip install -r requirements-core.txt

# Create necessary directories
echo ""
echo "Step 4: Creating project directories..."
mkdir -p config rec log models metrics

# Make scripts executable
echo "Making scripts executable..."
chmod +x bin/voice_transcriber.py bin/whisper_daemon.py bin/setup_backend.py bin/tui_app.py

echo ""
echo "=================================================="
echo "  ✓ Core Installation Complete!"
echo "=================================================="
echo ""
echo "Installed with:"
echo "  ✓ Core dependencies (audio, TUI, utilities)"
echo "  ✓ Vosk backend (lightweight, fast - perfect for 4GB RAM)"
echo "  ✗ Whisper backend (NOT installed - requires 900MB download)"
echo ""
echo "Next Steps:"
echo ""
echo "1. Setup your backend (will download Vosk model ~40MB):"
echo "   source venv/bin/activate"
echo "   python bin/setup_backend.py --interactive"
echo ""
echo "2. Launch the TUI dashboard:"
echo "   python bin/voice_transcriber.py --tui"
echo ""
echo "Optional: Install Whisper backend (for high accuracy):"
echo "   ./install-whisper.sh"
echo "   (WARNING: Downloads 900MB, may timeout on slow connection)"
echo ""
echo "For Sway hotkey setup, see: WAYLAND_HOTKEY_GUIDE.md"
echo ""