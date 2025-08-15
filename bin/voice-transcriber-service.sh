#!/bin/bash
"""
Voice Transcriber Service Wrapper Script
Handles virtual environment activation and environment setup for systemd service.
Ensures proper Wayland/Hyprland integration and logging.
"""

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$SCRIPT_DIR/voice_transcriber.py"
LOG_DIR="$PROJECT_DIR/log"
SERVICE_LOG="$LOG_DIR/service.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to log messages with timestamp
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$SERVICE_LOG"
    # Also send to systemd journal
    echo "[$level] $message" | systemd-cat -t voice-transcriber
}

# Function to handle cleanup on exit
cleanup() {
    log_message "INFO" "Service shutdown requested, cleaning up..."
    # Kill any child processes
    if [ -n "$PYTHON_PID" ]; then
        log_message "INFO" "Terminating voice transcriber process (PID: $PYTHON_PID)"
        kill -TERM "$PYTHON_PID" 2>/dev/null
        wait "$PYTHON_PID" 2>/dev/null
    fi
    log_message "INFO" "Voice transcriber service stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Start service
log_message "INFO" "Starting Voice Transcriber Service"
log_message "INFO" "Project directory: $PROJECT_DIR"
log_message "INFO" "Virtual environment: $VENV_PATH"
log_message "INFO" "Python script: $PYTHON_SCRIPT"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log_message "ERROR" "Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_message "ERROR" "Voice transcriber script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Source environment setup script
if [ -f "$SCRIPT_DIR/setup-wayland-environment.sh" ]; then
    log_message "INFO" "Loading Wayland/Hyprland environment"
    source "$SCRIPT_DIR/setup-wayland-environment.sh"
else
    log_message "WARNING" "Wayland environment setup script not found"
fi

# Activate virtual environment
log_message "INFO" "Activating Python virtual environment"
source "$VENV_PATH/bin/activate"

if [ $? -ne 0 ]; then
    log_message "ERROR" "Failed to activate virtual environment"
    exit 1
fi

# Verify Python and required modules
log_message "INFO" "Python version: $(python --version 2>&1)"
log_message "INFO" "Python executable: $(which python)"

# Check for required Python modules
python -c "import whisper, pyaudio, pynput" 2>/dev/null
if [ $? -ne 0 ]; then
    log_message "ERROR" "Required Python modules not available"
    log_message "INFO" "Please run: pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Wait for audio system to be ready
log_message "INFO" "Waiting for audio system to be ready..."
sleep 2

# Check if audio system is available
if command -v pactl >/dev/null 2>&1; then
    if ! pactl info >/dev/null 2>&1; then
        log_message "WARNING" "PulseAudio/PipeWire not ready, waiting..."
        sleep 5
    fi
fi

# Start the voice transcriber
log_message "INFO" "Starting voice transcriber application"
log_message "INFO" "Command: python $PYTHON_SCRIPT"

# Start Python script in background and capture PID
python "$PYTHON_SCRIPT" &
PYTHON_PID=$!

log_message "INFO" "Voice transcriber started with PID: $PYTHON_PID"

# Wait for the Python process to complete
wait "$PYTHON_PID"
EXIT_CODE=$?

log_message "INFO" "Voice transcriber process exited with code: $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE