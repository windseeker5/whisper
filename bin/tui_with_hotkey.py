#!/usr/bin/env python3
"""
TUI with Hotkey Support - Exactly like your original version but with TUI display

Features:
- Listens for SUPER+A hotkey (no Sway config needed!)
- Audio feedback (beep on start/stop)
- Shows transcription in TUI
- Auto-copies to clipboard
"""

import os
import sys
import time
import threading
import signal
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'bin'))


class HotkeyTUI:
    """TUI that listens for hotkey and shows transcriptions."""

    def __init__(self, app):
        """Initialize with WhisperApp instance."""
        self.app = app
        self.running = True
        self.display_lock = threading.Lock()
        self.last_transcription = ""
        self.transcription_history = []

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def play_sound(self, sound_type: str):
        """Play audio feedback (start/stop beep)."""
        try:
            if sound_type == 'start':
                # Higher pitched beep for start
                os.system('paplay /usr/share/sounds/freedesktop/stereo/message.oga 2>/dev/null &')
            elif sound_type == 'stop':
                # Lower pitched beep for stop
                os.system('paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null &')
        except:
            # Fallback to system beep
            print('\a', end='', flush=True)

    def display(self):
        """Display current status and transcriptions."""
        with self.display_lock:
            self.clear_screen()

            # Header
            backend = self.app.config.get('backend', 'vosk').upper()
            status = "🔴 RECORDING" if self.app.audio_processor.is_recording else "⚫ READY"

            print("=" * 70)
            print(f"  Voice Transcriber - {backend} Backend")
            print(f"  Status: {status}")
            print(f"  Hotkey: SUPER+A (start/stop recording)")
            print("=" * 70)

            # Current transcription
            if self.last_transcription:
                print("\n┌─ Latest Transcription " + "─" * 45 + "┐")
                # Word wrap
                words = self.last_transcription.split()
                line = ""
                for word in words:
                    if len(line + word) > 65:
                        print(f"│ {line:<66} │")
                        line = word + " "
                    else:
                        line += word + " "
                if line.strip():
                    print(f"│ {line.strip():<66} │")
                print("└" + "─" * 68 + "┘")
                print("✓ Copied to clipboard")
            else:
                print("\n[Waiting for recording... Press SUPER+A to start]")

            # History
            if self.transcription_history:
                print("\n─── Recent History ───")
                for item in self.transcription_history[-5:]:
                    timestamp = item['timestamp'].strftime("%H:%M:%S")
                    text = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
                    print(f"  [{timestamp}] {text}")

            # Footer
            print("\n" + "─" * 70)
            print("  Press Ctrl+C to quit")
            print("=" * 70)

    def on_recording_start(self):
        """Called when recording starts."""
        self.play_sound('start')
        self.display()

    def on_recording_stop(self, transcription: str):
        """Called when recording stops and transcription is ready."""
        self.last_transcription = transcription

        # Add to history
        self.transcription_history.append({
            'text': transcription,
            'timestamp': datetime.now()
        })

        # Keep only last 20
        self.transcription_history = self.transcription_history[-20:]

        # Copy to clipboard
        try:
            import pyperclip
            pyperclip.copy(transcription)
        except:
            pass

        # Play stop sound
        self.play_sound('stop')

        # Update display
        self.display()

    def run(self):
        """Main display loop."""
        # Initial display
        self.display()

        # Keep running and updating display when needed
        try:
            while self.running:
                time.sleep(0.5)

                # Update display if recording state changed
                # (The app handles the actual recording)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.running = False


def run_tui_with_hotkey(app):
    """Run TUI with hotkey support.

    Args:
        app: WhisperApp instance with audio processor and transcriber
    """
    tui = HotkeyTUI(app)

    # Monkey-patch the app to notify TUI
    original_toggle = app.toggle_recording

    def toggle_recording_with_tui():
        """Wrapper that notifies TUI."""
        was_recording = app.audio_processor.is_recording

        if not was_recording:
            # Starting recording
            tui.on_recording_start()

        # Call original toggle
        result = original_toggle()

        if was_recording:
            # Just stopped recording - get transcription
            # The transcription is in the result or we can get it from app
            if hasattr(app, 'last_transcription') and app.last_transcription:
                tui.on_recording_stop(app.last_transcription)

        return result

    app.toggle_recording = toggle_recording_with_tui

    # Run TUI
    tui.run()


if __name__ == "__main__":
    print("This module should be imported by voice_transcriber.py")
