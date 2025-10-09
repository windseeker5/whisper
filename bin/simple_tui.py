#!/usr/bin/env python3
"""
Simple TUI for Voice Transcription - No fancy graphics, just works!

A minimal terminal interface that's reliable and fast.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'bin'))


class SimpleTUI:
    """Simple text-based interface - no fancy graphics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = None
        self.running = True
        self.is_recording = False
        self.current_transcription = ""
        self.transcription_history = []

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def load_backend(self):
        """Load transcription backend."""
        from transcription_backends import BackendFactory

        try:
            self.backend = BackendFactory.create_backend(self.config)
            success = self.backend.load_model()

            info = self.backend.get_model_info()
            if success:
                print(f"✓ {info['backend']} backend loaded: {info.get('model', 'unknown')}")
                return True
            else:
                print(f"✗ Failed to load {info['backend']} backend")
                return False
        except Exception as e:
            print(f"✗ Error loading backend: {e}")
            return False

    def show_header(self):
        """Show simple header."""
        backend_name = self.backend.get_backend_name() if self.backend else "Unknown"
        status = "🔴 RECORDING" if self.is_recording else "⚫ IDLE"

        print("=" * 60)
        print(f"  Voice Transcriber - {backend_name} Backend")
        print(f"  Status: {status}")
        print("=" * 60)

    def show_menu(self):
        """Show menu options."""
        print("\nCommands:")
        print("  r - Toggle recording")
        print("  c - Copy last transcription")
        print("  h - Show history")
        print("  f - Show files")
        print("  b - Switch backend")
        print("  q - Quit")
        print()

    def show_transcription(self):
        """Show current transcription."""
        if self.current_transcription:
            print("\n┌─ Last Transcription ───────────────────────────────┐")
            # Wrap long text
            words = self.current_transcription.split()
            line = ""
            for word in words:
                if len(line + word) > 50:
                    print(f"│ {line:<50} │")
                    line = word + " "
                else:
                    line += word + " "
            if line.strip():
                print(f"│ {line.strip():<50} │")
            print("└────────────────────────────────────────────────────┘")
        else:
            print("\n[No transcription yet]")

    def show_history(self):
        """Show transcription history."""
        self.clear_screen()
        print("\n=== Transcription History ===\n")

        if not self.transcription_history:
            print("No history yet.")
        else:
            for i, item in enumerate(self.transcription_history[-10:], 1):
                timestamp = item['timestamp'].strftime("%H:%M:%S")
                text = item['text'][:60] + "..." if len(item['text']) > 60 else item['text']
                print(f"{i}. [{timestamp}] {text}")

        input("\nPress Enter to continue...")

    def show_files(self):
        """Show recording files."""
        self.clear_screen()
        print("\n=== Recording Files ===\n")

        rec_dir = Path(self.config.get('recordings_dir', 'rec'))
        if not rec_dir.exists():
            print("No recordings yet.")
        else:
            files = sorted(rec_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)

            if not files:
                print("No recordings yet.")
            else:
                for i, file_path in enumerate(files[:20], 1):
                    stat = file_path.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    date = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"{i}. {file_path.name:<30} {size_mb:>6.2f} MB  {date}")

                print(f"\nTotal: {len(files)} files")

        input("\nPress Enter to continue...")

    def toggle_recording(self):
        """Simulate recording toggle."""
        if not self.is_recording:
            print("\n🎤 Recording started... (This is a demo - real recording not implemented yet)")
            self.is_recording = True

            # Simulate recording for 2 seconds
            time.sleep(2)

            self.is_recording = False
            print("⏹  Recording stopped. Processing...")

            # Simulate transcription
            demo_text = f"Demo transcription at {datetime.now().strftime('%H:%M:%S')}"
            self.current_transcription = demo_text
            self.transcription_history.append({
                'text': demo_text,
                'timestamp': datetime.now()
            })

            print(f"✓ Done: {demo_text}")
        else:
            print("Already recording...")

    def copy_transcription(self):
        """Copy current transcription to clipboard."""
        if not self.current_transcription:
            print("Nothing to copy")
            return

        try:
            import pyperclip
            pyperclip.copy(self.current_transcription)
            print("✓ Copied to clipboard!")
        except Exception as e:
            print(f"✗ Copy failed: {e}")

    def switch_backend(self):
        """Switch backend."""
        current = self.config.get('backend', 'vosk')
        new_backend = 'whisper' if current == 'vosk' else 'vosk'

        print(f"\nSwitching from {current} to {new_backend}...")
        self.config['backend'] = new_backend

        if self.backend:
            self.backend.unload_model()

        self.load_backend()
        print("✓ Backend switched!")
        time.sleep(1)

    def run(self):
        """Main loop."""
        self.clear_screen()
        print("Loading...")

        if not self.load_backend():
            print("\nPress Enter to exit...")
            input()
            return

        while self.running:
            self.clear_screen()
            self.show_header()
            self.show_transcription()
            self.show_menu()

            try:
                cmd = input("Command: ").strip().lower()

                if cmd == 'r':
                    self.toggle_recording()
                    time.sleep(1)
                elif cmd == 'c':
                    self.copy_transcription()
                    time.sleep(1)
                elif cmd == 'h':
                    self.show_history()
                elif cmd == 'f':
                    self.show_files()
                elif cmd == 'b':
                    self.switch_backend()
                elif cmd == 'q':
                    print("\nGoodbye!")
                    self.running = False
                else:
                    print(f"Unknown command: {cmd}")
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                self.running = False
            except Exception as e:
                print(f"\nError: {e}")
                time.sleep(2)


def run_simple_tui(config: Dict[str, Any]):
    """Run the simple TUI."""
    tui = SimpleTUI(config)
    tui.run()


if __name__ == "__main__":
    # Test
    test_config = {
        'backend': 'vosk',
        'vosk_model_path': './models/vosk-model-small-en-us-0.15',
        'recordings_dir': 'rec',
        'language': 'en'
    }

    run_simple_tui(test_config)
