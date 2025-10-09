#!/usr/bin/env python3
"""
Direct Keyboard Listener using evdev (works on Wayland without compositor config)

Listens for SUPER+A key combination directly from keyboard device.
Requires user to be in 'input' group.
"""

import threading
import logging
from typing import Callable, Optional


class KeyboardListener:
    """Listen for SUPER+A using evdev (direct keyboard access)."""

    def __init__(self, callback: Callable):
        """Initialize keyboard listener.

        Args:
            callback: Function to call when SUPER+A is pressed
        """
        self.callback = callback
        self.running = False
        self.thread = None
        self.super_pressed = False

    def find_keyboard_device(self):
        """Find the keyboard device."""
        try:
            from evdev import InputDevice, list_devices, ecodes

            # Find keyboard device
            devices = [InputDevice(path) for path in list_devices()]

            for device in devices:
                # Look for device with KEY_A capability (likely a keyboard)
                caps = device.capabilities()
                if ecodes.EV_KEY in caps:
                    keys = caps[ecodes.EV_KEY]
                    # Check if it has both SUPER and A keys
                    if ecodes.KEY_LEFTMETA in keys and ecodes.KEY_A in keys:
                        logging.info(f"Found keyboard: {device.name}")
                        return device

            return None

        except ImportError:
            logging.error("evdev not installed. Install with: pip install evdev")
            return None
        except Exception as e:
            logging.error(f"Error finding keyboard: {e}")
            return None

    def listen(self):
        """Main listening loop."""
        try:
            from evdev import ecodes, categorize, KeyEvent

            device = self.find_keyboard_device()
            if not device:
                logging.error("No keyboard device found")
                logging.info("Make sure you're in the 'input' group:")
                logging.info("  sudo usermod -a -G input $USER")
                logging.info("  (then log out and back in)")
                return

            logging.info(f"Listening for SUPER+A on {device.name}")

            for event in device.read_loop():
                if not self.running:
                    break

                if event.type == ecodes.EV_KEY:
                    key_event = categorize(event)

                    # Track SUPER key state
                    if key_event.keycode in ['KEY_LEFTMETA', 'KEY_RIGHTMETA']:
                        self.super_pressed = (key_event.keystate == KeyEvent.key_down)

                    # Check for A key press while SUPER is held
                    if key_event.keycode == 'KEY_A' and key_event.keystate == KeyEvent.key_down:
                        if self.super_pressed:
                            logging.info("SUPER+A detected!")
                            self.callback()

        except PermissionError:
            logging.error("Permission denied accessing keyboard")
            logging.info("Add yourself to input group:")
            logging.info("  sudo usermod -a -G input $USER")
            logging.info("  (then log out and back in)")
        except Exception as e:
            logging.error(f"Keyboard listener error: {e}")

    def start(self):
        """Start listening in background thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        logging.info("Keyboard listener started")

    def stop(self):
        """Stop listening."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logging.info("Keyboard listener stopped")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    def on_hotkey():
        print("SUPER+A pressed!")

    listener = KeyboardListener(on_hotkey)

    print("Starting keyboard listener...")
    print("Press SUPER+A to test (Ctrl+C to quit)")

    listener.start()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()
