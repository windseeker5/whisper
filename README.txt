===============================================================================
                          WHISPER VOICE-TO-TEXT APPLICATION
                             For Arch Linux with Hyprland
===============================================================================

OVERVIEW
--------
This is a voice-to-text application that uses OpenAI Whisper for transcription.
Press SUPER+A to start/stop recording, and your speech will be automatically
transcribed and copied to your clipboard.

FEATURES
--------
- Global hotkey (SUPER+A) for recording control
- Voice Activity Detection and noise reduction  
- Automatic transcription using OpenAI Whisper
- Clipboard integration and desktop notifications
- Configurable microphone and language settings
- Timestamped logging of all transcriptions
- Background daemon mode for always-on operation

WHAT'S IN THIS PROJECT
----------------------
Essential files only (cleaned up and simplified):

/bin/voice_transcriber.py  - Main application (run this)
/bin/whisper_daemon.py     - Background daemon service
/config/whisper_config.json - Your settings (created on first run)
/requirements.txt          - Python dependencies
/install.sh               - Installation script
/venv/                    - Python virtual environment
/rec/                     - Audio recordings stored here
/log/                     - Transcription logs stored here


SYSTEM REQUIREMENTS
-------------------
- Arch Linux with Hyprland (Wayland)
- Python 3.8 or newer
- Microphone access
- Internet connection (for initial Whisper model download)


INSTALLATION (Step-by-Step)
---------------------------

Step 1: Install System Dependencies
Run this command to install required system packages:

    sudo pacman -S ffmpeg libnotify pulseaudio portaudio python-pip

Step 2: Run Installation Script
From the whisper project directory, run:

    chmod +x install.sh
    ./install.sh

This will:
- Create a Python virtual environment
- Install all Python dependencies
- Create necessary directories
- Make scripts executable

Step 3: Configure the Application
Activate the virtual environment and run the configuration:

    source venv/bin/activate
    python bin/voice_transcriber.py --config

This will guide you through:
- Selecting your microphone from available devices
- Choosing Whisper model size (base recommended for most users)
- Setting language preference (auto-detect recommended)
- Adjusting audio sensitivity and gain settings

Your configuration is saved in config/whisper_config.json


HYPRLAND KEYBINDING SETUP
-------------------------

To enable the SUPER+A hotkey in Hyprland, add this line to your Hyprland config:

    ~/.config/hypr/hyprland.conf

Add the following keybinding:

    # Whisper Voice-to-Text Application
    bind = $mainMod, A, exec, /home/kdresdell/Documents/DEV/whisper/bin/voice_transcriber.py

Replace the path with your actual whisper project location.

After adding this line:
1. Save the config file
2. Restart Hyprland or reload the config: hyprctl reload
3. Test the SUPER+A hotkey

Note: If you're using a daemon setup, bind to the daemon toggle instead:
    bind = $mainMod, A, exec, pkill -SIGUSR1 -f whisper_daemon


DAILY USAGE
-----------

Option 1: Hotkey Mode (Recommended)
First, set up the Hyprland keybinding (see section above).

Activate the virtual environment and start the application:

    source venv/bin/activate  
    python bin/voice_transcriber.py

The app runs in the background. Press SUPER+A to start recording,
speak, then press SUPER+A again to stop. Your transcription will 
be copied to clipboard and shown as a notification.

Option 2: Manual Mode (For Testing)
If hotkeys don't work, use manual mode:

    source venv/bin/activate
    python bin/voice_transcriber.py --manual

Press Enter to start recording, speak, then Enter again to stop.

Option 3: Background Daemon
For always-on operation, run as a background daemon:

    source venv/bin/activate
    python bin/whisper_daemon.py start

Send signals to control:
- SIGUSR1: Toggle recording
- SIGUSR2: Show status  
- SIGTERM: Stop daemon

Example: pkill -SIGUSR1 -f whisper_daemon


CONFIGURATION OPTIONS
----------------------
Run configuration anytime to adjust settings:

    source venv/bin/activate
    python bin/voice_transcriber.py --config

Available settings:
- Microphone device selection
- Whisper model (tiny/base/small/medium/large) 
- Language (auto-detect/English/French/Spanish/German)
- Audio sensitivity and gain control
- Auto Gain Control (AGC) settings
- System microphone boost
- Audio processing options (normalization, compression)
- Real-time audio level display


AUDIO TROUBLESHOOTING
---------------------

If audio isn't working:

1. Test audio levels first:
    python bin/voice_transcriber.py --test-audio
   This shows real-time audio levels to help adjust sensitivity.

2. Reconfigure with audio compatibility detection:
    python bin/voice_transcriber.py --config
   The configuration detects compatible settings automatically.

3. Check system audio:
    - Ensure your microphone is working: pavucontrol
    - Test with: arecord -f cd -d 5 test.wav && aplay test.wav
    - Check permissions: user should be in 'audio' group

4. Try different devices:
    The configuration shows all compatible audio devices.


FILES AND DIRECTORIES
---------------------

/bin/voice_transcriber.py - Main application with full features
/bin/whisper_daemon.py    - Lightweight background daemon  
/config/                  - Configuration files (JSON format)
/rec/                     - Audio recordings (timestamped WAV files)
/log/                     - Transcription logs (daily text files)
/venv/                    - Python virtual environment (don't modify)


COMMAND-LINE OPTIONS
--------------------

python bin/voice_transcriber.py [OPTIONS]

  --config         Configure microphone, model, and audio settings
  --manual         Run in manual mode (press Enter to record)
  --test-audio     Test audio levels and sensitivity
  --daemon         Run as background daemon
  --help           Show detailed help and examples

For daemon control:
python bin/whisper_daemon.py [start|stop|restart|status|toggle]


PERFORMANCE NOTES
------------------

Model sizes and performance:
- tiny: Fastest, least accurate (~39 MB)
- base: Good balance, recommended (~74 MB) 
- small: Better accuracy (~244 MB)
- medium: High accuracy (~769 MB)
- large: Best accuracy (~1550 MB)

First transcription takes longer (model loading). Subsequent 
transcriptions are faster.

Memory usage:
- Idle: ~50MB
- With model loaded: ~200-800MB (depending on model size)


TIPS FOR BEST RESULTS
----------------------

1. Audio Quality:
   - Speak clearly and at normal volume
   - Minimize background noise
   - Use a good quality microphone
   - Adjust gain settings in configuration

2. Recording:
   - Start recording before speaking
   - Pause briefly before and after speech
   - Keep recordings under 30 seconds for best performance

3. Accuracy:
   - Use 'base' model for good balance of speed/accuracy
   - Set language specifically if not using English
   - Ensure good audio levels (use --test-audio to check)


TROUBLESHOOTING
---------------

Problem: Global hotkey (SUPER+A) doesn't work
Solution: Use manual mode: python bin/voice_transcriber.py --manual

Problem: No audio detected
Solution: Run audio test: python bin/voice_transcriber.py --test-audio
         Reconfigure: python bin/voice_transcriber.py --config

Problem: Poor transcription quality  
Solution: Check audio levels, reduce background noise, try larger model

Problem: Application crashes
Solution: Check log files in /log/ directory for error messages

Problem: Dependencies missing
Solution: Rerun installation: ./install.sh

Problem: Permission denied
Solution: Add user to audio group: sudo usermod -a -G audio $USER
         Then log out and back in.


SUPPORT AND LOGS
-----------------

Log files are automatically created in the /log/ directory:
- whisper_app.log: Application logs and errors
- transcriptions_YYYY-MM-DD.txt: Daily transcription history

Check these files for troubleshooting information.


VERSION INFORMATION
-------------------
Whisper Voice-to-Text Application v1.0.0
Compatible with: Arch Linux, Hyprland, Wayland
Python: 3.8+, OpenAI Whisper, PyAudio

===============================================================================