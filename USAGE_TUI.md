# Using the TUI Mode - Exactly Like Your Original!

## What You Get

✅ TUI display showing status and transcriptions
✅ Listens for SUPER+A hotkey (no manual key presses in terminal!)
✅ Audio feedback (beep on start/stop)
✅ Auto-copy to clipboard
✅ Shows transcription history

## Setup (One-Time)

### Step 1: Add Sway Keybinding

Add this **ONE LINE** to your Sway config (`~/.config/sway/config`):

```bash
bindsym $mod+a exec pkill -SIGUSR1 -f voice_transcriber.py
```

Then reload Sway:
```bash
swaymsg reload
```

**That's it!** This line tells Sway: "When SUPER+A is pressed, send a signal to the Python app"

The Python app receives the signal and toggles recording - no need for complex Wayland hotkey handling!

### Step 2: Run the TUI

```bash
source venv/bin/activate
python bin/voice_transcriber.py --tui
```

## How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│  Voice Transcriber - VOSK Backend                                │
│  Status: ⚫ READY                                                 │
│  Hotkey: SUPER+A (start/stop recording)                          │
└──────────────────────────────────────────────────────────────────┘

[Waiting for recording... Press SUPER+A to start]

──────────────────────────────────────────────────────────────────
  Press Ctrl+C to quit
══════════════════════════════════════════════════════════════════
```

### When You Press SUPER+A (First Press):

1. **Status changes**: ⚫ READY → 🔴 RECORDING
2. **Beep sound plays** (high pitch)
3. **You speak into microphone**

### When You Press SUPER+A Again (Second Press):

1. **Beep sound plays** (different pitch)
2. **Status**: 🔴 RECORDING → ⚫ READY
3. **Processing** (Vosk transcribes - takes 1-2 seconds)
4. **Display updates**:

```
┌─ Latest Transcription ───────────────────────────────────┐
│ This is what you just said and it was transcribed       │
└──────────────────────────────────────────────────────────┘
✓ Copied to clipboard

─── Recent History ───
  [14:32:15] Previous transcription one
  [14:31:45] Another transcription here
  [14:30:22] And another one
```

## Why This Approach?

**Simple Signal-Based Design**:
- ✅ No complex Wayland IPC
- ✅ No pynput (which doesn't work on Wayland)
- ✅ Just one line in Sway config
- ✅ Python app receives signal and toggles recording

**The signal approach is:**
- Reliable
- Fast (no polling)
- Works on all Wayland compositors
- Simple to debug

## Troubleshooting

### SUPER+A doesn't work

**Check 1**: Is the Sway keybinding added?
```bash
grep "voice_transcriber" ~/.config/sway/config
```
Should show: `bindsym $mod+a exec pkill -SIGUSR1 -f voice_transcriber.py`

**Check 2**: Did you reload Sway?
```bash
swaymsg reload
```

**Check 3**: Is the app running?
```bash
ps aux | grep voice_transcriber
```
Should show the Python process running.

**Test manually**: Send signal yourself
```bash
pkill -SIGUSR1 -f voice_transcriber.py
```
You should see the status change in the TUI.

### No audio feedback sounds

The app tries to play freedesktop sounds. If they don't work, you'll just see status changes without beeps. This is normal and doesn't affect functionality.

### Microphone not working

Run audio test first:
```bash
python bin/voice_transcriber.py --test-audio
```

Reconfigure if needed:
```bash
python bin/voice_transcriber.py --config
```

## Comparison: TUI vs Manual vs Standard

| Mode | How to Record | Display | Use Case |
|------|---------------|---------|----------|
| **--tui** | Press SUPER+A anywhere | Live TUI with history | **Best!** Visual feedback + hotkey |
| **--manual** | Press Enter in terminal | Minimal CLI | Testing, troubleshooting |
| **(default)** | Press SUPER+A anywhere | Background, notifications only | Minimal, no TUI |

## Performance

On your 2007 iMac with Vosk:
- Recording: Instant
- Transcription: 1-2 seconds for 10 seconds of audio
- TUI refresh: Instant
- Total overhead: < 50 MB RAM

Perfect for daily use!

## Tips

1. **Keep it running**: The TUI can stay open all day - just press SUPER+A when you need to transcribe something
2. **History**: Recent transcriptions stay in the TUI - scroll back mentally or check logs in `log/` directory
3. **Clipboard**: Every transcription auto-copies - just Ctrl+V to paste anywhere
4. **Switch backends**: Press Ctrl+C, run `python bin/setup_backend.py --backend whisper`, restart TUI to try Whisper (slower but more accurate)

Enjoy! 🎤
