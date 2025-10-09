# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a voice-to-text transcription application for Arch Linux with Wayland/Hyprland that supports multiple transcription backends (Whisper and Vosk). The application provides hotkey-activated voice recording with automatic transcription, clipboard integration, desktop notifications, and an optional interactive TUI dashboard.

**Key Feature**: Press SUPER+A to toggle recording; transcription is automatically copied to clipboard.

**New TUI Mode**: Full-screen terminal dashboard with 3 panels (audio levels, transcription output, file manager) plus backend switching.

## Architecture

### Core Components

The application consists of multiple executables and a backend abstraction layer:

**Main Executables:**
1. **voice_transcriber.py** - Main application with CLI and TUI modes, configuration wizard, audio diagnostics
2. **whisper_daemon.py** - Lightweight background daemon with lazy model loading
3. **tui_app.py** - Interactive TUI dashboard (launched via `--tui` flag)

**Backend System** (`transcription_backends.py`):
- `TranscriptionBackend` - Abstract base class for all backends
- `WhisperBackend` - OpenAI Whisper implementation (high accuracy, slow on old hardware)
- `VoskBackend` - Vosk implementation (fast, lightweight, perfect for 4GB RAM systems)
- `BackendFactory` - Factory pattern for instantiating backends

**Shared Classes** (defined in `voice_transcriber.py`):
- `WhisperConfig` - JSON-based configuration management
- `AudioProcessor` - PyAudio recording with auto-detection of PulseAudio/PipeWire/ALSA
- `DesktopIntegration` - Clipboard (pyperclip) and notifications (libnotify)
- `TranscriptionLogger` - Timestamped logging to daily files

### Wayland/Hyprland Integration

**Critical**: Standard global hotkey libraries (pynput) don't work reliably on Wayland due to security restrictions.

**Solution**: The application uses signal-based triggering:
- Hyprland keybind sends SIGUSR1 to the process: `bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py`
- Application handles signal to toggle recording state
- Fallback: Named pipe at `/tmp/whisper_hotkey_trigger`

Files:
- `bin/wayland_hotkey_handler.py` - Multi-strategy hotkey handler with Hyprland IPC, signals, and named pipes
- `bin/setup_hyprland_hotkey.sh` - Automation script for Hyprland configuration

### Audio System Detection

The application automatically detects and configures for:
- **PipeWire** (most common on modern Arch)
- **PulseAudio** (legacy)
- **ALSA** (fallback)

`SimpleAudioDetector` class runs detection on startup and configures PyAudio accordingly. Audio configuration is validated on launch and auto-corrected if incompatible.

### Service Management

Systemd user service for auto-start on login:
- `voice-transcriber.service` - systemd unit file
- `bin/voice-transcriber-service.sh` - wrapper script with environment setup
- `bin/manage-voice-service.py` - CLI management tool (install/start/stop/logs)
- `bin/setup-wayland-environment.sh` - Sets Wayland-specific environment variables

## Installation & Setup

### Initial Installation
```bash
# Install system dependencies
sudo pacman -S ffmpeg libnotify pulseaudio portaudio python-pip

# Run installation script (creates venv, installs Python deps)
./install.sh

# Configure application (microphone selection, model size, language)
source venv/bin/activate
python bin/voice_transcriber.py --config
```

### Hyprland Integration
Add to `~/.config/hypr/hyprland.conf`:
```bash
# Voice transcription hotkey
bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py

# Optional: Auto-start on login
exec-once = cd /home/kdresdell/DEV/whisper && source venv/bin/activate && python bin/voice_transcriber.py &
```

### Systemd Service Setup
```bash
# Install and enable service for auto-start
python bin/manage-voice-service.py install

# Service control
python bin/manage-voice-service.py start|stop|restart|status|logs

# Alternative: use systemctl directly
systemctl --user start voice-transcriber.service
```

## Common Commands

### Backend Setup (First Time)
```bash
# Interactive backend setup wizard
source venv/bin/activate
python bin/setup_backend.py --interactive

# Or set backend directly
python bin/setup_backend.py --backend vosk   # Recommended for 4GB RAM
python bin/setup_backend.py --backend whisper

# List available backends
python bin/setup_backend.py --list
```

### Running the Application
```bash
source venv/bin/activate

# TUI mode - Interactive dashboard with 3 panels (RECOMMENDED)
python bin/voice_transcriber.py --tui

# Standard CLI mode (with hotkey support)
python bin/voice_transcriber.py

# Manual mode (press Enter to toggle recording)
python bin/voice_transcriber.py --manual

# Configuration wizard
python bin/voice_transcriber.py --config

# Audio diagnostics
python bin/voice_transcriber.py --test-audio
```

### Daemon Mode
```bash
# Start daemon (lazy model loading, minimal memory footprint)
python bin/whisper_daemon.py start

# Toggle recording via signal
pkill -SIGUSR1 -f whisper_daemon.py

# Check status
pkill -SIGUSR2 -f whisper_daemon.py

# Stop daemon
python bin/whisper_daemon.py stop
```

### Diagnostic Tools
```bash
# Comprehensive audio system diagnostics
python bin/audio_diagnostics.py

# Test silence removal algorithm
python bin/test_silence_removal.py

# Test audio quality metrics
python bin/test_audio_quality.py

# Test hotkey detection
python bin/test_hotkey.py

# Performance monitoring
python bin/performance_monitor.py
```

## Configuration

Configuration file: `config/whisper_config.json`

Key settings:
- `backend` - Transcription backend: "vosk" or "whisper"
- `microphone_device` - PyAudio device identifier (e.g., "pyaudio:0")
- `language` - "auto" for auto-detect, or specific: "en"/"fr"/"es"/etc.
- `sample_rate` - Audio sample rate in Hz (16000 recommended)
- `advanced_silence_removal` - Fine-tuned silence detection for cleaner recordings

**Backend-Specific Settings:**
- Whisper: `whisper_model` - Model size: tiny/base/small/medium/large
- Vosk: `vosk_model_path` - Path to Vosk model directory (e.g., "./models/vosk-model-small-en-us-0.15")

The config is auto-validated on startup. If audio settings are incompatible, the application will auto-detect and update to working values.

## TUI Dashboard

When running with `--tui`, you get a full-screen terminal interface with:

**Panel 1 - Audio Levels (Top)**
- Real-time animated audio meter with color zones
- Recording status indicator
- Peak level and gain display

**Panel 2 - Transcription Output (Middle)**
- Current transcription display with copy/paste
- Recent history viewer (last 20 transcriptions)
- Export and clear buttons

**Panel 3 - File Manager (Bottom)**
- Table view of all recordings
- Play, delete, bulk cleanup actions
- Storage usage indicator

**Sidebar - Settings & Control**
- Backend switcher (hot-swap between Vosk/Whisper)
- Model selector
- Performance metrics (CPU, RAM, transcription time)
- Quick settings

**Keyboard Shortcuts:**
- `R` - Toggle recording
- `C` - Copy current transcription
- `D` - Delete selected file
- `B` - Switch backend
- `Q` - Quit
- `Tab` - Navigate panels

## Directory Structure

```
/bin/                                      - Executable scripts and tools
  voice_transcriber.py                     - Main application (CLI and TUI modes)
  whisper_daemon.py                        - Background daemon service
  tui_app.py                               - TUI dashboard (3-panel interface)
  transcription_backends.py                - Backend abstraction layer
  setup_backend.py                         - Backend setup wizard
  wayland_hotkey_handler.py                - Wayland/Hyprland hotkey integration
  manage-voice-service.py                  - Systemd service management
  audio_diagnostics.py                     - Audio system diagnostics
  french_transcription_optimizer.py        - French language optimization
  performance_monitor.py                   - Resource usage monitoring
  test_*.py                                - Testing utilities

/config/                                   - Configuration files (JSON)
  whisper_config.json                      - Main application config (created on first run)

/models/                                   - Vosk models (downloaded by setup script)
  vosk-model-small-en-us-0.15/             - Example: Small English model (~40MB)

/rec/                                      - Recorded audio files (timestamped WAV)
/log/                                      - Application and transcription logs
/metrics/                                  - Performance metrics (JSON)
/venv/                                     - Python virtual environment
```

## Backend Comparison: Vosk vs Whisper

### Vosk (Recommended for 4GB RAM / Old Hardware)

**Pros:**
- ✓ **Very fast**: 10-20x faster than Whisper on old CPUs
- ✓ **Low RAM**: 50-200 MB depending on model size
- ✓ **Offline**: No internet required after model download
- ✓ **Small models**: 40-50 MB for basic models
- ✓ **Perfect for 2007 iMac with 4GB RAM**

**Cons:**
- ✗ Lower accuracy than Whisper (still quite good)
- ✗ Fewer language-specific optimizations

**Performance on 2007 iMac:**
- 10 seconds audio → ~1-2 seconds transcription ✓

### Whisper (Recommended for 6GB+ RAM / Modern Hardware)

**Pros:**
- ✓ **High accuracy**: Best-in-class transcription quality
- ✓ **Language support**: Excellent for 99+ languages
- ✓ **Robust**: Handles accents, noise, multiple speakers well
- ✓ **French optimization**: Special handling for French phonetics

**Cons:**
- ✗ **Slow on old hardware**: 30-120 seconds for 10s audio on 2007 iMac
- ✗ **High RAM**: 500MB-2GB depending on model
- ✗ **Large models**: 74MB (base) to 1550MB (large)

**Performance on 2007 iMac:**
- 10 seconds audio → 30-120 seconds transcription ✗ (painfully slow)

## Model Information

### Whisper Models
- **tiny** (~39 MB): Fastest, lowest accuracy, good for testing
- **base** (~74 MB): **Recommended for Whisper** - best speed/accuracy balance
- **small** (~244 MB): Better accuracy, moderate speed
- **medium** (~769 MB): High accuracy, slower
- **large** (~1550 MB): Best accuracy, slowest

Models are downloaded on first use and cached in `~/.cache/whisper/`.

### Vosk Models
- **vosk-model-small-en-us-0.15** (~40 MB): **Recommended for 4GB RAM** - Fast, lightweight
- **vosk-model-en-us-0.22** (~1.8 GB): Larger, more accurate English model
- **vosk-model-small-fr-0.22** (~41 MB): French support

Download from: https://alphacephei.com/vosk/models
Or use `python bin/setup_backend.py --interactive` to download automatically.

## Audio Processing Pipeline

1. **Recording**: PyAudio captures audio at configured sample rate
2. **Level Monitoring**: Real-time audio level display (optional)
3. **Gain Control**: Microphone gain boost (configurable)
4. **Silence Removal**: Advanced algorithm removes leading/trailing/internal silence
5. **Normalization**: Optional audio normalization
6. **Transcription**: Whisper processes cleaned audio
7. **Post-processing**: Text copied to clipboard, notification shown, log written

## Troubleshooting

### Hotkey Not Working
1. Verify Hyprland config: `hyprctl binds | grep SUPER`
2. Test signal manually: `pkill -SIGUSR1 -f voice_transcriber.py`
3. Check application is running: `ps aux | grep voice_transcriber`
4. Fallback: Use `--manual` mode

### Audio Issues
1. Run diagnostics: `python bin/audio_diagnostics.py`
2. Test audio levels: `python bin/voice_transcriber.py --test-audio`
3. Reconfigure: `python bin/voice_transcriber.py --config`
4. Check system: `pactl info` (PulseAudio/PipeWire status)

### Service Issues
```bash
# Check service status
python bin/manage-voice-service.py status

# View logs
python bin/manage-voice-service.py logs
journalctl --user -u voice-transcriber.service -f

# Restart service
python bin/manage-voice-service.py restart
```

## Development Notes

### Adding New Features

When modifying core functionality:
1. Main application logic is in `voice_transcriber.py` (classes are reused by daemon)
2. Daemon-specific optimizations are in `whisper_daemon.py`
3. Update configuration schema in `WhisperConfig` class
4. Test with both interactive and daemon modes
5. Verify Wayland/Hyprland signal handling still works

### Audio Backend Changes

Audio system detection is in `SimpleAudioDetector` class. If adding support for new audio systems:
1. Add new `AudioSystem` enum value
2. Update `detect_audio_system()` method
3. Add backend-specific initialization in `_initialize_audio_backend()`
4. Test with `audio_diagnostics.py`

### Signal Handling

The application uses Unix signals for IPC:
- `SIGUSR1` - Toggle recording (main control signal)
- `SIGUSR2` - Status query
- `SIGTERM` - Graceful shutdown
- `SIGINT` - Keyboard interrupt handler

Signal handlers are registered in both `voice_transcriber.py` and `whisper_daemon.py`.

## Performance Considerations

- **Daemon lazy loading**: Whisper model loads only on first transcription (startup < 1 second)
- **Memory footprint**: ~50MB idle, ~200-800MB with model loaded (varies by model size)
- **Audio processing**: Real-time gain control and level monitoring are optional (disable for lower overhead)
- **Silence removal**: Advanced algorithm reduces transcription time and improves accuracy
- **Metrics**: Performance data logged to `metrics/transcription_metrics_*.json`

## French Language Support

Special tool for French transcription optimization: `bin/french_transcription_optimizer.py`
- Tests different model sizes for French accuracy
- Benchmarks performance vs quality trade-offs
- Analyzes French phonetic preprocessing

Usage: `python bin/french_transcription_optimizer.py --test-models`
