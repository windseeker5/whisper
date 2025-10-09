# Installation Guide

## Quick Install (Recommended for 4GB RAM)

This installs **only Vosk backend** - no torch download, no timeouts!

```bash
cd /home/kdresdell/DEV/whisper

# 1. Install system dependencies
sudo pacman -S ffmpeg libnotify portaudio

# 2. Run install script
./install.sh

# 3. Setup Vosk backend (downloads 40MB model)
source venv/bin/activate
python bin/setup_backend.py --interactive

# 4. Launch TUI
python bin/voice_transcriber.py --tui
```

**Total download size**: ~100 MB (no torch!)

---

## Installing Whisper Backend (Optional)

⚠️ **WARNING**: Whisper requires downloading torch (~888 MB) which may timeout on slow connections.

**Only install if:**
- You have 6GB+ RAM
- You need maximum accuracy
- You have a reliable internet connection

### Method 1: Auto-install with resume support

```bash
./install-whisper.sh
```

This script uses `wget` to download torch, which supports resuming if it times out. Just run the script again and it will continue from where it left off.

### Method 2: Manual download (most reliable)

```bash
# 1. Download torch manually with wget (supports resume)
wget -c https://files.pythonhosted.org/packages/16/82/3948e54c01b2109238357c6f86242e6ecbf0c63a1af46906772902f82057/torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl

# 2. Install torch from local file
source venv/bin/activate
pip install torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl

# 3. Install whisper and dependencies
pip install -r requirements-whisper.txt

# 4. Clean up
rm torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl

# 5. Configure whisper backend
python bin/setup_backend.py --backend whisper
```

If download times out, just run the `wget -c` command again - it will resume!

### Method 3: CPU-only torch (much smaller)

If you don't need GPU support, install CPU-only version:

```bash
source venv/bin/activate

# Install CPU-only torch (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install whisper
pip install openai-whisper librosa

# Configure
python bin/setup_backend.py --backend whisper
```

---

## Troubleshooting

### Pip install times out

**Problem**: `error: incomplete-download` when installing requirements.txt

**Solution**: Use the new installation method which skips whisper/torch:
```bash
pip install -r requirements-core.txt  # Only installs Vosk + core deps
```

### Torch download fails repeatedly

**Solution**: Use wget method (see Method 2 above) - it supports resume!

### System dependencies missing

```bash
# Check what's missing
pacman -Q ffmpeg libnotify portaudio

# Install missing packages
sudo pacman -S ffmpeg libnotify portaudio
```

### Virtual environment issues

```bash
# Remove and recreate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

---

## What Gets Installed

### Core Installation (requirements-core.txt)
- PyAudio, sounddevice, soundfile (audio recording)
- pydub, numpy, scipy (audio processing)
- pynput, pyperclip (hotkeys, clipboard)
- textual, rich (TUI interface)
- **vosk** (fast transcription backend)
- psutil (performance monitoring)

**Total**: ~100 MB

### Whisper Installation (requirements-whisper.txt)
- **torch** (~888 MB - the problematic one!)
- openai-whisper
- librosa

**Total**: ~900 MB additional

---

## Verification

After installation, verify everything works:

```bash
source venv/bin/activate

# Check backends
python bin/setup_backend.py --list

# Test backend
python -c "import vosk; print('Vosk OK')"
python -c "import whisper; print('Whisper OK')"  # Only if you installed it

# Launch TUI
python bin/voice_transcriber.py --tui
```

---

## File Structure After Install

```
/home/kdresdell/DEV/whisper/
├── bin/                    # Scripts
├── config/                 # Config files (created on first run)
├── log/                    # Log files
├── models/                 # Vosk models (downloaded by setup script)
├── rec/                    # Recordings
├── venv/                   # Python virtual environment
├── install.sh              # Main installer (Vosk only)
├── install-whisper.sh      # Optional Whisper installer
├── requirements-core.txt   # Core + Vosk deps (no torch!)
├── requirements-whisper.txt # Whisper deps (includes torch)
└── QUICKSTART.md           # Quick start guide
```
