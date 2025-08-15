# Wayland/Hyprland Hotkey Configuration Guide

This guide explains how to resolve hotkey issues with the Whisper Voice Transcriber on Wayland compositors, specifically Hyprland.

## Problem Analysis

The original error messages you see:
```
ERROR:root:Error setting up hotkey: <super>
Warning: Global hotkey may not work properly on Wayland.
You can still use the application by running with --manual mode.
```

These occur because:

1. **Pynput Wayland Incompatibility**: The `pynput` library was designed for X11 and has limited Wayland support
2. **Security Model**: Wayland restricts global input monitoring for security reasons
3. **Key Format Issues**: The `<super>` key specification may not be properly recognized

## Solutions Implemented

### 1. Enhanced Hotkey Handler (`wayland_hotkey_handler.py`)

The new handler provides multiple fallback strategies:
- **Hyprland IPC integration** for native compositor support
- **Named pipe communication** for reliable triggering
- **Improved pynput fallback** with alternative key combinations
- **Signal-based manual mode** for maximum compatibility

### 2. Updated Voice Transcriber

The main application now:
- **Auto-detects Wayland sessions** and uses appropriate handlers
- **Provides better error messages** with actionable solutions
- **Supports multiple triggering methods** including signals and pipes

### 3. Hyprland Configuration Script

The `setup_hyprland_hotkey.sh` script configures:
- **SUPER+A** keybind using process signals
- **SUPER+SHIFT+A** backup using named pipes
- **Automatic setup** and cleanup instructions

## Quick Setup

1. **Run the setup script:**
   ```bash
   cd /home/kdresdell/Documents/DEV/whisper
   ./bin/setup_hyprland_hotkey.sh
   ```

2. **Start the voice transcriber:**
   ```bash
   source venv/bin/activate
   python bin/voice_transcriber.py
   ```

3. **Test the hotkey:**
   - Press `SUPER+A` to toggle recording
   - If that doesn't work, try `SUPER+SHIFT+A`

## Alternative Methods

### Method 1: Signal-Based (Recommended)

Add to your Hyprland config (`~/.config/hypr/hyprland.conf`):
```bash
bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py
```

### Method 2: Named Pipe
```bash
bind = SUPER, a, exec, echo 'trigger' > /tmp/whisper_hotkey_trigger
```

### Method 3: Manual Mode
```bash
python bin/voice_transcriber.py --manual
```

## Troubleshooting

### Issue: Hotkey not responding
**Solutions:**
1. Check if the voice transcriber process is running: `ps aux | grep voice_transcriber`
2. Verify Hyprland keybind is set: `hyprctl binds | grep SUPER`
3. Test signal manually: `pkill -SIGUSR1 -f voice_transcriber.py`

### Issue: Permission errors with named pipe
**Solutions:**
1. Check pipe permissions: `ls -la /tmp/whisper_hotkey_trigger`
2. Recreate the pipe: `rm /tmp/whisper_hotkey_trigger && mkfifo /tmp/whisper_hotkey_trigger`
3. Set proper permissions: `chmod 666 /tmp/whisper_hotkey_trigger`

### Issue: Hyprland keybind conflicts
**Solutions:**
1. Check existing binds: `hyprctl binds | grep "SUPER, a"`
2. Use alternative key: `SUPER+SHIFT+A` or `ALT+SHIFT+A`
3. Unbind conflicting keys: `hyprctl keyword unbind SUPER, a`

## Testing the Setup

1. **Start the application:**
   ```bash
   cd /home/kdresdell/Documents/DEV/whisper
   source venv/bin/activate
   python bin/voice_transcriber.py
   ```

2. **Check logs:**
   ```bash
   tail -f log/whisper_app.log
   ```

3. **Test signal method:**
   ```bash
   pkill -SIGUSR1 -f voice_transcriber.py
   ```

4. **Test pipe method:**
   ```bash
   echo 'trigger' > /tmp/whisper_hotkey_trigger
   ```

## Making Keybinds Permanent

Add these lines to your Hyprland configuration file (`~/.config/hypr/hyprland.conf`):

```bash
# Whisper Voice Transcriber hotkeys
bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py
bind = SUPER_SHIFT, a, exec, echo 'trigger' > /tmp/whisper_hotkey_trigger

# Optional: Start voice transcriber on Hyprland startup
exec-once = cd /home/kdresdell/Documents/DEV/whisper && source venv/bin/activate && python bin/voice_transcriber.py &
```

Then reload Hyprland configuration:
```bash
hyprctl reload
```

## Performance Considerations

- **Signal method** is fastest and most reliable
- **Named pipe** has minimal overhead
- **Pynput fallback** may have higher latency on Wayland
- **Manual mode** has no background overhead

## Security Notes

- Signal-based triggering only affects processes you own
- Named pipes are created with appropriate permissions
- No global input monitoring is required
- All methods respect Wayland's security model

## Additional Resources

- [Hyprland Keybind Documentation](https://wiki.hyprland.org/Configuring/Binds/)
- [Wayland Security Model](https://wayland.freedesktop.org/docs/html/ch04.html#sect-Protocol-Security)
- [Pynput Wayland Issues](https://github.com/moses-palmer/pynput/issues)