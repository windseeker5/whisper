#!/bin/bash
# Setup script for Hyprland hotkey integration with Whisper Voice Transcriber
# This script configures SUPER+A hotkey to work with the voice transcriber on Hyprland/Wayland

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$WHISPER_DIR/venv"
TRANSCRIBER_SCRIPT="$SCRIPT_DIR/voice_transcriber.py"

echo "Setting up Hyprland hotkey for Whisper Voice Transcriber..."

# Check if we're running on Hyprland
if [ -z "$HYPRLAND_INSTANCE_SIGNATURE" ]; then
    echo "Warning: HYPRLAND_INSTANCE_SIGNATURE not found. Are you running Hyprland?"
    echo "This script is designed for Hyprland compositor."
    exit 1
fi

# Check if the voice transcriber exists
if [ ! -f "$TRANSCRIBER_SCRIPT" ]; then
    echo "Error: Voice transcriber script not found at $TRANSCRIBER_SCRIPT"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run install.sh first to set up the environment."
    exit 1
fi

echo "Configuring Hyprland keybind..."

# Method 1: Direct process signal (recommended)
echo "Setting up SUPER+A to send SIGUSR1 signal to voice_transcriber.py process"
hyprctl keyword bind "SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py || echo 'Voice transcriber not running'"

# Method 2: Named pipe approach (alternative)
echo "Setting up named pipe trigger method as backup"
PIPE_PATH="/tmp/whisper_hotkey_trigger"

# Create the pipe if it doesn't exist
if [ ! -p "$PIPE_PATH" ]; then
    mkfifo "$PIPE_PATH"
    chmod 666 "$PIPE_PATH"
fi

# Add alternative bind that uses the pipe
hyprctl keyword bind "SUPER_SHIFT, a, exec, echo 'trigger' > $PIPE_PATH || echo 'Pipe trigger failed'"

echo "Hotkey configuration complete!"
echo ""
echo "Configured keybindings:"
echo "  SUPER+A        - Send signal to running voice transcriber"
echo "  SUPER+SHIFT+A  - Trigger via named pipe (backup method)"
echo ""
echo "To start the voice transcriber:"
echo "  cd $WHISPER_DIR"
echo "  source venv/bin/activate"
echo "  python bin/voice_transcriber.py"
echo ""
echo "To remove the keybindings:"
echo "  hyprctl keyword unbind SUPER, a"
echo "  hyprctl keyword unbind SUPER_SHIFT, a"
echo ""
echo "Note: These keybindings are temporary and will be lost on Hyprland restart."
echo "To make them permanent, add them to your Hyprland configuration file:"
echo ""
echo "  bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py"
echo "  bind = SUPER_SHIFT, a, exec, echo 'trigger' > /tmp/whisper_hotkey_trigger"