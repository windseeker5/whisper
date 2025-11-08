#!/bin/bash
# Launcher script for Whisper TUI
# This script activates the virtual environment and launches the TUI

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment
source venv/bin/activate

# Launch TUI
exec python bin/voice_transcriber.py --tui
