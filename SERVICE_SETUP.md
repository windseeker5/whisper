# Voice Transcriber Service Automation Setup

This guide helps you set up the voice transcriber as an automated systemd user service on Arch Linux with Hyprland.

## Features

- **Auto-start on login**: Service starts automatically when you log into your Hyprland session
- **Proper environment setup**: Automatically configures Wayland/Hyprland environment variables
- **Virtual environment handling**: Automatically activates the Python virtual environment
- **Comprehensive logging**: Service and application logs for troubleshooting
- **Easy management**: Simple commands to start, stop, restart, and monitor the service

## Quick Setup

1. **Install the service:**
   ```bash
   cd /home/kdresdell/Documents/DEV/whisper
   python bin/manage-voice-service.py install
   ```

2. **Start the service:**
   ```bash
   python bin/manage-voice-service.py start
   ```

3. **Check service status:**
   ```bash
   python bin/manage-voice-service.py status
   ```

## Service Management Commands

### Installation and Setup
```bash
# Install and enable service for auto-start
python bin/manage-voice-service.py install
```

### Service Control
```bash
# Start the service
python bin/manage-voice-service.py start

# Stop the service
python bin/manage-voice-service.py stop

# Restart the service
python bin/manage-voice-service.py restart

# Check service status
python bin/manage-voice-service.py status
```

### Monitoring and Troubleshooting
```bash
# View recent logs (last 50 lines)
python bin/manage-voice-service.py logs

# View more log lines
python bin/manage-voice-service.py logs --log-lines 100

# Monitor logs in real-time
journalctl --user -u voice-transcriber.service -f
```

### Uninstallation
```bash
# Remove and disable the service
python bin/manage-voice-service.py uninstall
```

## Alternative systemctl Commands

You can also use standard systemctl commands:

```bash
# Service control
systemctl --user start voice-transcriber.service
systemctl --user stop voice-transcriber.service
systemctl --user restart voice-transcriber.service
systemctl --user status voice-transcriber.service

# Enable/disable auto-start
systemctl --user enable voice-transcriber.service
systemctl --user disable voice-transcriber.service

# View logs
journalctl --user -u voice-transcriber.service
```

## Log Files

The service creates logs in multiple locations:

1. **Systemd journal**: `journalctl --user -u voice-transcriber.service`
2. **Service log**: `/home/kdresdell/Documents/DEV/whisper/log/service.log`
3. **Application log**: `/home/kdresdell/Documents/DEV/whisper/log/whisper_app.log`

## Troubleshooting

### Service won't start
1. Check dependencies:
   ```bash
   python bin/manage-voice-service.py status
   ```

2. Verify virtual environment:
   ```bash
   source venv/bin/activate
   python -c "import whisper, pyaudio, pynput"
   ```

3. Check audio system:
   ```bash
   pactl info
   ```

### Hotkey not working
1. Ensure Hyprland is running and detected
2. Check Wayland environment variables in logs
3. Verify the hotkey handler is properly initialized

### Audio issues
1. Check microphone permissions
2. Verify audio device configuration in `config/whisper_config.json`
3. Test audio recording manually

## Files Created

The automation setup creates these files:

- `/home/kdresdell/Documents/DEV/whisper/voice-transcriber.service` - systemd service definition
- `/home/kdresdell/Documents/DEV/whisper/bin/voice-transcriber-service.sh` - service wrapper script
- `/home/kdresdell/Documents/DEV/whisper/bin/setup-wayland-environment.sh` - environment setup
- `/home/kdresdell/Documents/DEV/whisper/bin/manage-voice-service.py` - service management tool
- `~/.config/systemd/user/voice-transcriber.service` - installed service file (symlink)

## Security Features

The service runs with enhanced security:
- No new privileges
- Protected system directories
- Restricted file system access
- Private temporary directory

## Auto-start Behavior

Once installed and enabled, the service will:
1. Start automatically when you log into Hyprland
2. Wait for the desktop environment to be ready
3. Set up proper Wayland/Hyprland environment variables
4. Activate the Python virtual environment
5. Start the voice transcriber with hotkey support

The service will restart automatically if it crashes (up to 3 times within 30 seconds).