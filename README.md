# Whisper Voice Transcription

A simple voice-to-text application for Arch Linux with Wayland/Hyprland. Press SUPER+A to record, transcription automatically copied to clipboard.

## Features

- **Hotkey Recording** - SUPER+A to toggle recording
- **Multiple Backends** - Whisper (high accuracy) or Vosk (fast, low RAM)
- **Built-in TUI** - Interactive terminal dashboard
- **Clipboard Integration** - Auto-copy transcriptions
- **Desktop Notifications** - Visual feedback

## Quick Start

### 1. Install Dependencies

```bash
# System packages
sudo pacman -S ffmpeg libnotify pulseaudio portaudio python-pip

# Python dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
source venv/bin/activate
python bin/voice_transcriber.py --config
```

This wizard lets you:
- Select your microphone
- Choose backend (Whisper or Vosk)
- Pick model size
- Set language

### 3. Run

**TUI Mode (Recommended):**
```bash
python bin/voice_transcriber.py --tui
```

**Standard CLI Mode:**
```bash
python bin/voice_transcriber.py
```

**Manual Mode (Press Enter to record):**
```bash
python bin/voice_transcriber.py --manual
```

## Hyprland Setup

Add to `~/.config/hypr/hyprland.conf`:

```bash
# Voice transcription hotkey
bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py

# Optional: Auto-start on login
exec-once = cd /path/to/whisper && source venv/bin/activate && python bin/voice_transcriber.py &
```

Reload Hyprland config or restart Hyprland.

## Backend Comparison

### Whisper (Currently Configured)
- **Accuracy**: Excellent
- **Speed**: Moderate to slow on old hardware
- **RAM**: 500MB - 2GB depending on model
- **Best for**: Modern systems, accuracy-critical tasks

### Vosk
- **Accuracy**: Good
- **Speed**: Very fast (10-20x faster than Whisper)
- **RAM**: 50-200MB
- **Best for**: 4GB RAM systems, old hardware, speed matters

Switch backends: `python bin/voice_transcriber.py --config`

## Project Structure

```
whisper/
├── bin/
│   ├── voice_transcriber.py          # Main application
│   └── transcription_backends.py     # Backend support
├── config/
│   └── whisper_config.json           # Your settings
├── models/                           # Vosk models (if using Vosk)
├── rec/                              # Recorded audio files
├── log/                              # Application logs
└── requirements.txt                  # Python dependencies
```

## Troubleshooting

**Test your microphone:**
```bash
python bin/voice_transcriber.py --test-audio
```

**Reconfigure everything:**
```bash
python bin/voice_transcriber.py --config
```

**Check logs:**
```bash
tail -f log/whisper_app.log
```

**Hotkey not working:**
1. Check Hyprland config: `hyprctl binds | grep SUPER`
2. Test manually: `pkill -SIGUSR1 -f voice_transcriber.py`
3. Use `--manual` mode as fallback

## License

MIT License - See individual component licenses for dependencies.

---

**Made for Arch Linux with Wayland/Hyprland**
