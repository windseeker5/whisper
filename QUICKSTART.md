# Quick Start Guide - Voice Transcriber with TUI

## What's New

Your voice transcription app now has:

✨ **Multiple Backends**: Choose between Whisper (accurate) or Vosk (fast, perfect for your 4GB RAM iMac)
✨ **Interactive TUI Dashboard**: Beautiful 3-panel interface with live monitoring
✨ **Easy Setup**: Guided backend installation wizard

## Installation (Fixed for Timeout Issues!)

### 1. Install Core + Vosk (NO torch download, NO timeouts!)

```bash
cd /home/kdresdell/DEV/whisper

# Install system dependencies
sudo pacman -S ffmpeg libnotify portaudio

# Run main installer (installs Vosk only - lightweight!)
./install.sh
```

**This installs:**
- ✓ Core dependencies (~50 MB)
- ✓ Vosk backend (~3 MB library + 40 MB model)
- ✓ TUI interface
- ✗ Whisper/torch (skipped to avoid timeout)

**Total download**: ~100 MB - Quick and reliable!

### 2. Setup Vosk Backend

```bash
source venv/bin/activate
python bin/setup_backend.py --interactive
```

This will:
- Detect your system specs (4GB RAM)
- Recommend **Vosk** for your hardware
- Download the Vosk model (~40 MB)
- Configure everything automatically

### 3. (Optional) Install Whisper Backend

**Only if you want maximum accuracy and have time for the large download:**

```bash
# Method 1: Auto-install with resume support
./install-whisper.sh

# Method 2: Manual download (if script fails)
wget -c https://files.pythonhosted.org/packages/16/82/3948e54c01b2109238357c6f86242e6ecbf0c63a1af46906772902f82057/torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl
pip install torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl
pip install -r requirements-whisper.txt
rm torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl
```

**wget supports resume!** If it times out, just run the same `wget -c` command again.

## Running the App

### Option 1: TUI Mode (RECOMMENDED)

```bash
python bin/voice_transcriber.py --tui
```

**You'll see:**
```
┌─────────────────────────────────────────────────────┐
│ Audio Levels                                        │
│ ● RECORDING                                         │
│ Level: [████████░░░░░░░░░░] -15.2 dB               │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│ Transcriptions                                      │
│ This is your transcribed text...                    │
│ [Copy] [Clear] [Export]                            │
│                                                     │
│ Recent History:                                     │
│ 14:32:15 This is a test transcription              │
│ 14:30:42 Hello world example                       │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│ Recording Files                                     │
│ File              Date       Size    Duration       │
│ rec_143215.wav   14:32:15   2.3MB   00:15          │
│ [Delete] [Play] [Refresh] [Clean Old]             │
└─────────────────────────────────────────────────────┘
```

**Controls:**
- `R` - Start/Stop recording
- `C` - Copy current transcription
- `D` - Delete selected file
- `B` - Switch backend (Vosk ⇄ Whisper)
- `Q` - Quit

### Option 2: Standard CLI Mode

```bash
# With hotkey (SUPER+A)
python bin/voice_transcriber.py

# Manual mode (press Enter to record)
python bin/voice_transcriber.py --manual
```

## Performance on Your 2007 iMac

**With Vosk (recommended):**
- 5 second recording → ~1 second transcription ✓ FAST
- RAM usage: ~150 MB ✓ LOW
- CPU usage: Moderate

**With Whisper (if you installed it):**
- 5 second recording → ~30-60 seconds transcription ✗ SLOW
- RAM usage: ~600 MB ✗ HIGH
- CPU usage: 100% (maxed out)

**Recommendation**: Stick with Vosk for daily use!

## Switching Backends

You can switch backends anytime:

**In TUI mode**: Press `B` key

**Via CLI**:
```bash
python bin/setup_backend.py --backend whisper  # Switch to Whisper
python bin/setup_backend.py --backend vosk     # Switch back to Vosk
```

**Via config file**: Edit `config/whisper_config.json` and change:
```json
{
  "backend": "vosk"
}
```

## Troubleshooting

### TUI won't launch

```bash
# Install TUI dependencies
pip install textual rich
```

### Vosk model not found

```bash
# Run setup again
python bin/setup_backend.py --interactive
```

### Hotkey doesn't work

Add to your Sway config (`~/.config/sway/config`):
```bash
bindsym $mod+a exec pkill -SIGUSR1 -f voice_transcriber.py
```

Then reload Sway: `swaymsg reload`

## File Locations

- **Configuration**: `config/whisper_config.json`
- **Recordings**: `rec/` (WAV files)
- **Logs**: `log/whisper_app.log`
- **Vosk models**: `models/vosk-model-small-en-us-0.15/`

## Next Steps

1. **Try the TUI**: `python bin/voice_transcriber.py --tui`
2. **Test recording**: Press `R`, speak for 5 seconds, press `R` again
3. **Copy text**: Press `C` to copy transcription to clipboard
4. **Manage files**: Use the file manager panel to delete old recordings

## Getting Help

- Run configuration wizard: `python bin/voice_transcriber.py --config`
- Test audio: `python bin/voice_transcriber.py --test-audio`
- List backends: `python bin/setup_backend.py --list`
- Check logs: `tail -f log/whisper_app.log`

Enjoy your new voice transcription app! 🎤
