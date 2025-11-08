# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

A voice-to-text transcription application for Arch Linux with Wayland/Hyprland. Press SUPER+A to toggle recording; transcription is automatically copied to clipboard. Supports Whisper (high accuracy) and Vosk (fast, low RAM) backends with built-in TUI dashboard.

## Architecture

### Core Components

**Main Application:**
- `bin/voice_transcriber.py` - Complete application with CLI, TUI, configuration wizard, and audio processing

**Backend System:**
- `bin/transcription_backends.py` - Backend abstraction layer
  - `TranscriptionBackend` - Abstract base class
  - `WhisperBackend` - OpenAI Whisper implementation
  - `VoskBackend` - Vosk implementation
  - `BackendFactory` - Factory for instantiating backends

**Key Classes** (in voice_transcriber.py):
- `WhisperConfig` - JSON-based configuration management
- `AudioProcessor` - PyAudio recording with auto-detection of PulseAudio/PipeWire/ALSA
- `DesktopIntegration` - Clipboard and notifications
- `TranscriptionLogger` - Timestamped logging
- `HotkeyListeningTUI` - Built-in TUI dashboard
- `WhisperApp` - Main application controller

### Wayland/Hyprland Integration

Standard hotkey libraries don't work on Wayland. Solution: signal-based triggering via Hyprland keybind that sends SIGUSR1:

```bash
bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py
```

### Audio System Detection

Auto-detects and configures for PipeWire, PulseAudio, or ALSA. `SimpleAudioDetector` class runs on startup.

## Configuration

File: `config/whisper_config.json`

Key settings:
- `backend` - "vosk" or "whisper"
- `microphone_device` - PyAudio device (e.g., "pyaudio:0")
- `language` - "auto" or specific: "en"/"fr"/"es"/etc.
- `sample_rate` - Audio sample rate (16000 recommended)
- `whisper_model` - Whisper: tiny/base/small/medium/large
- `vosk_model_path` - Vosk: path to model directory

## Common Commands

### Running the App

```bash
# TUI mode (interactive dashboard) - RECOMMENDED
python bin/voice_transcriber.py --tui

# Standard CLI mode
python bin/voice_transcriber.py

# Manual mode (press Enter to toggle)
python bin/voice_transcriber.py --manual

# Configuration wizard
python bin/voice_transcriber.py --config

# Audio test
python bin/voice_transcriber.py --test-audio
```

### Backend Setup

Use the configuration wizard to switch backends:
```bash
python bin/voice_transcriber.py --config
```

## Backend Comparison

### Vosk (Recommended for 4GB RAM)
- Very fast (10-20x faster than Whisper)
- Low RAM: 50-200 MB
- Small models: 40-50 MB
- Perfect for old hardware
- Good accuracy

### Whisper (Recommended for 6GB+ RAM)
- Best accuracy
- Excellent multi-language support
- Slow on old hardware
- 500MB-2GB RAM
- Models: 39MB (tiny) to 1550MB (large)

## Directory Structure

```
whisper/
├── bin/
│   ├── voice_transcriber.py          # Main application (3300 lines)
│   └── transcription_backends.py     # Backend abstraction (400 lines)
├── config/
│   └── whisper_config.json           # Configuration
├── models/                           # Vosk models (downloaded by user)
├── rec/                              # Recorded audio files (WAV)
├── log/                              # Application logs
├── metrics/                          # Performance metrics (JSON)
├── venv/                             # Python virtual environment
├── README.md                         # User documentation
└── requirements.txt                  # Python dependencies
```

## Development Notes

### Adding Features

Core logic is in `voice_transcriber.py`. The file is self-contained with all classes inline:
1. Audio system detection classes (AudioSystem, AudioDevice, SimpleAudioDetector)
2. Core application classes (WhisperConfig, AudioProcessor, etc.)
3. TUI implementation (HotkeyListeningTUI)
4. Main application (WhisperApp)
5. CLI entry point (main function)

### Signal Handling

Unix signals for IPC:
- `SIGUSR1` - Toggle recording (main control)
- `SIGUSR2` - Status query
- `SIGTERM` - Graceful shutdown
- `SIGINT` - Keyboard interrupt

### Audio Processing Pipeline

1. PyAudio captures audio at configured sample rate
2. Real-time audio level monitoring (optional)
3. Gain control and boost
4. Advanced silence removal
5. Audio normalization (optional)
6. Transcription via selected backend
7. Copy to clipboard, show notification, write log

## Troubleshooting

### Hotkey Issues
1. Test signal: `pkill -SIGUSR1 -f voice_transcriber.py`
2. Check Hyprland: `hyprctl binds | grep SUPER`
3. Fallback: `--manual` mode

### Audio Issues
1. Test audio: `python bin/voice_transcriber.py --test-audio`
2. Reconfigure: `python bin/voice_transcriber.py --config`
3. Check system: `pactl info`

### View Logs
```bash
tail -f log/whisper_app.log
```

## Performance

- Memory footprint: ~50-200MB (Vosk) or ~500MB-2GB (Whisper)
- Transcription time: 1-2s for 10s audio (Vosk) or 2-30s (Whisper, hardware dependent)
- Silence removal improves accuracy and speed
- Real-time audio level display is optional

## Technical Notes

- The app is fully self-contained in voice_transcriber.py (except backend abstraction)
- Built-in TUI uses basic terminal manipulation (no external TUI library)
- Configuration is auto-validated and corrected on startup
- Backend selection is runtime-configurable
- Works on Wayland/Hyprland via signal-based hotkeys
- No daemon mode in simplified version - app runs in foreground
