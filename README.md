# Voice Transcription App with TUI Dashboard

A voice-to-text transcription application for Arch Linux with Sway/Hyprland, featuring multiple backends (Vosk/Whisper) and a beautiful terminal UI.

## Features

🎤 **Voice Recording** - Hotkey-activated (SUPER+A) or manual recording
🤖 **Multiple Backends** - Choose Vosk (fast, 4GB RAM) or Whisper (accurate, 6GB+ RAM)
💻 **Interactive TUI** - 3-panel dashboard with audio levels, transcriptions, and file manager
📋 **Clipboard Integration** - Auto-copy transcriptions
🔔 **Desktop Notifications** - Visual feedback on completion
🎯 **Wayland/Sway Support** - Signal-based hotkeys work on Wayland

## Screenshots

### TUI Mode
```
┌─ Audio Levels ──────────────────────────────────────┐
│ ● RECORDING                                         │
│ [████████████░░░░░░░░] -12.5 dB                    │
└─────────────────────────────────────────────────────┘
┌─ Transcriptions ────────────────────────────────────┐
│ This is your transcribed text...                    │
│ [Copy] [Clear] [Export]                            │
└─────────────────────────────────────────────────────┘
┌─ Recording Files ───────────────────────────────────┐
│ rec_143215.wav   14:32:15   2.3MB   00:15          │
│ [Delete] [Play] [Refresh]                          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install (NO timeout issues - uses lightweight Vosk)
./install.sh

# 2. Setup backend
source venv/bin/activate
python bin/setup_backend.py --interactive

# 3. Launch TUI
python bin/voice_transcriber.py --tui
```

👉 **See [QUICKSTART.md](QUICKSTART.md) for detailed instructions**

## Installation Options

### Option 1: Vosk Only (Recommended for 4GB RAM)
- **Download**: ~100 MB total
- **Speed**: Very fast (1-2 seconds for 10s audio)
- **RAM**: ~150 MB
- **Perfect for**: 2007 iMac, old hardware, limited RAM

```bash
./install.sh
```

### Option 2: Add Whisper (For Maximum Accuracy)
- **Download**: +900 MB (torch)
- **Speed**: Slow on old hardware (30-120s for 10s audio)
- **RAM**: ~600 MB - 2 GB
- **Perfect for**: Modern systems, when accuracy matters

```bash
./install-whisper.sh  # Uses wget with resume support
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick getting started guide
- **[INSTALL.md](INSTALL.md)** - Detailed installation instructions
- **[CLAUDE.md](CLAUDE.md)** - Technical documentation
- **[WAYLAND_HOTKEY_GUIDE.md](WAYLAND_HOTKEY_GUIDE.md)** - Hotkey setup for Wayland
- **[SERVICE_SETUP.md](SERVICE_SETUP.md)** - Systemd service setup

## Usage

### TUI Mode (Recommended)
```bash
python bin/voice_transcriber.py --tui
```

**Controls**: `R`=Record, `C`=Copy, `D`=Delete, `B`=Switch Backend, `Q`=Quit

### CLI Modes
```bash
# Standard mode with hotkey (SUPER+A)
python bin/voice_transcriber.py

# Manual mode (press Enter to record)
python bin/voice_transcriber.py --manual

# Configuration wizard
python bin/voice_transcriber.py --config
```

### Daemon Mode
```bash
# Start background service
python bin/whisper_daemon.py start

# Toggle recording via signal
pkill -SIGUSR1 -f whisper_daemon.py
```

## System Requirements

### Minimum (Vosk backend)
- 4GB RAM
- Dual-core CPU (Core 2 Duo or newer)
- 500 MB disk space
- Arch Linux with Sway/Hyprland

### Recommended (Whisper backend)
- 8GB RAM
- Quad-core CPU (2015 or newer)
- 2 GB disk space

## Architecture

```
bin/
├── voice_transcriber.py          # Main app (CLI + TUI)
├── tui_app.py                    # TUI dashboard
├── transcription_backends.py     # Backend abstraction
├── setup_backend.py              # Backend installer
└── whisper_daemon.py             # Background daemon

config/
└── whisper_config.json           # Configuration

models/
└── vosk-model-small-en-us-0.15/  # Vosk model
```

## Performance Comparison

### On 2007 iMac (4GB RAM, Core 2 Duo):

| Backend | 10s Audio | RAM Usage | Verdict |
|---------|-----------|-----------|---------|
| **Vosk** | ~1-2 seconds | ~150 MB | ✓ Excellent |
| **Whisper (tiny)** | ~30 seconds | ~400 MB | ⚠️ Slow |
| **Whisper (base)** | ~60 seconds | ~600 MB | ✗ Very slow |

### On Modern System (8GB+ RAM, 2020+ CPU):

| Backend | 10s Audio | RAM Usage | Verdict |
|---------|-----------|-----------|---------|
| **Vosk** | ~0.5 seconds | ~150 MB | ✓ Very fast |
| **Whisper (base)** | ~2-3 seconds | ~600 MB | ✓ Best quality |

## Troubleshooting

### Installation times out
Use the new split requirements:
```bash
pip install -r requirements-core.txt  # Vosk only, no torch
```

### Torch download fails
Use wget method (supports resume):
```bash
wget -c https://files.pythonhosted.org/packages/16/82/3948e54c01b2109238357c6f86242e6ecbf0c63a1af46906772902f82057/torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl
pip install torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl
```

### Hotkey doesn't work
Add to Sway config:
```bash
bindsym $mod+a exec pkill -SIGUSR1 -f voice_transcriber.py
```

### More Help
```bash
python bin/voice_transcriber.py --help
python bin/setup_backend.py --list
tail -f log/whisper_app.log
```

## Contributing

This is a personal project optimized for Arch Linux + Sway/Hyprland. Feel free to adapt for your needs!

## License

See individual component licenses. Main dependencies:
- Vosk - Apache 2.0
- OpenAI Whisper - MIT
- Textual - MIT

---

**Made for**: 2007 iMac with 4GB RAM running Arch + Sway
**Optimized for**: Lightweight operation with Vosk backend
**Works with**: Modern systems too (with either backend)
