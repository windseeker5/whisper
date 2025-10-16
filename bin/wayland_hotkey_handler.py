#!/home/kdresdell/DEV/whisper/venv/bin/python
"""
Wayland-compatible hotkey handler for Hyprland voice transcriber.

This module provides alternative hotkey implementations that work properly
on Wayland compositors, specifically Hyprland.
"""

import os
import subprocess
import threading
import time
import logging
from typing import Callable, Optional
import socket
import json


class HyprlandIPCHandler:
    """Handler for Hyprland IPC communication."""
    
    def __init__(self):
        self.hyprland_instance = os.environ.get('HYPRLAND_INSTANCE_SIGNATURE')
        self.socket_path = f"/tmp/hypr/{self.hyprland_instance}/.socket2.sock" if self.hyprland_instance else None
        self.running = False
        self.callback = None
        
    def is_available(self) -> bool:
        """Check if Hyprland IPC is available."""
        return (self.socket_path and 
                os.path.exists(self.socket_path) and 
                self.hyprland_instance is not None)
    
    def start_listening(self, callback: Callable[[], None]) -> bool:
        """Start listening for Hyprland events."""
        if not self.is_available():
            return False
            
        self.callback = callback
        self.running = True
        
        # Start listener thread
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
        return True
    
    def _listen_loop(self):
        """Listen for Hyprland IPC events."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.socket_path)
            
            while self.running:
                try:
                    data = sock.recv(1024)
                    if not data:
                        break
                        
                    # Parse Hyprland event
                    event_line = data.decode('utf-8').strip()
                    if 'activewindow>>' in event_line or 'workspace>>' in event_line:
                        # This is a simple implementation - you'd need to enhance
                        # this to detect specific key combinations
                        pass
                        
                except Exception as e:
                    logging.error(f"Error in Hyprland IPC listener: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logging.error(f"Failed to connect to Hyprland IPC: {e}")
        finally:
            try:
                sock.close()
            except:
                pass
    
    def stop(self):
        """Stop the IPC listener."""
        self.running = False


class WaylandHotkeyHandler:
    """Wayland-compatible hotkey handler with multiple fallback strategies."""

    def __init__(self):
        self.callback = None
        self.running = False
        self.method = None
        self.hyprland_handler = HyprlandIPCHandler()
        self.last_trigger_time = 0
        self.debounce_delay = 0.5  # 500ms debounce to prevent double-triggers
        
    def setup_hotkey(self, callback: Callable[[], None]) -> bool:
        """Setup hotkey with the best available method."""
        self.callback = callback
        
        # Try different methods in order of preference
        methods = [
            self._setup_hyprland_bind,
            self._setup_pynput_fallback,
            self._setup_manual_mode
        ]
        
        for method in methods:
            try:
                if method():
                    logging.info(f"Hotkey setup successful using {method.__name__}")
                    return True
            except Exception as e:
                logging.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        logging.error("All hotkey methods failed")
        return False
    
    def _setup_hyprland_bind(self) -> bool:
        """Setup hotkey using Hyprland's bind command."""
        try:
            # Create a named pipe for communication
            pipe_path = "/tmp/whisper_hotkey_trigger"
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            
            os.mkfifo(pipe_path)
            
            # Add Hyprland keybind that writes to the pipe
            bind_command = [
                "hyprctl", "keyword", "bind", 
                "SUPER, a, exec, echo 'trigger' > /tmp/whisper_hotkey_trigger"
            ]
            
            result = subprocess.run(bind_command, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to set Hyprland bind: {result.stderr}")
                return False
            
            # Start monitoring the pipe
            self.running = True
            self.method = "hyprland_bind"
            thread = threading.Thread(target=self._monitor_pipe, args=(pipe_path,), daemon=True)
            thread.start()
            
            logging.info("Hyprland hotkey bind set successfully")
            return True
            
        except Exception as e:
            logging.error(f"Hyprland bind setup failed: {e}")
            return False
    
    def _monitor_pipe(self, pipe_path: str):
        """Monitor the named pipe for hotkey triggers."""
        while self.running:
            try:
                with open(pipe_path, 'r') as pipe:
                    data = pipe.read().strip()
                    if data == 'trigger' and self.callback:
                        # Debouncing: only trigger if enough time has passed since last trigger
                        current_time = time.time()
                        if current_time - self.last_trigger_time >= self.debounce_delay:
                            self.last_trigger_time = current_time
                            self.callback()
                        else:
                            logging.debug(f"Ignored rapid hotkey press (debounced)")
            except Exception as e:
                logging.error(f"Pipe monitoring error: {e}")
                time.sleep(1)
    
    def _setup_pynput_fallback(self) -> bool:
        """Fallback to pynput with improved error handling."""
        try:
            from pynput import keyboard
            
            # Try different key combinations that might work better on Wayland
            key_combinations = [
                {'<cmd>+a': self.callback},  # Alternative super key representation
                {'<alt>+<shift>+a': self.callback},  # Fallback combination
            ]
            
            for combo in key_combinations:
                try:
                    hotkey_listener = keyboard.GlobalHotKeys(combo)
                    hotkey_listener.start()
                    self.method = "pynput"
                    logging.info(f"Pynput hotkey registered: {list(combo.keys())[0]}")
                    return True
                except Exception as e:
                    logging.warning(f"Failed to register {list(combo.keys())[0]}: {e}")
                    continue
            
            return False
            
        except ImportError:
            logging.error("Pynput not available")
            return False
    
    def _setup_manual_mode(self) -> bool:
        """Setup manual triggering mode."""
        logging.info("Falling back to manual mode")
        print("Hotkey setup failed. You can trigger recording by:")
        print("1. Running with --manual flag")
        print("2. Sending SIGUSR1 to the process")
        print("3. Using the Hyprland keybind if configured")

        # Setup signal handler for manual triggering with debouncing
        import signal
        def debounced_callback(sig, frame):
            if self.callback:
                current_time = time.time()
                if current_time - self.last_trigger_time >= self.debounce_delay:
                    self.last_trigger_time = current_time
                    self.callback()
                else:
                    logging.debug(f"Ignored rapid signal (debounced)")

        signal.signal(signal.SIGUSR1, debounced_callback)

        self.method = "manual"
        return True
    
    def cleanup(self):
        """Clean up hotkey handler."""
        self.running = False
        
        if self.method == "hyprland_bind":
            # Remove the Hyprland bind
            try:
                subprocess.run([
                    "hyprctl", "keyword", "unbind", "SUPER, a"
                ], capture_output=True)
                
                # Clean up pipe
                pipe_path = "/tmp/whisper_hotkey_trigger"
                if os.path.exists(pipe_path):
                    os.unlink(pipe_path)
            except Exception as e:
                logging.error(f"Cleanup error: {e}")


def create_wayland_hotkey_handler() -> WaylandHotkeyHandler:
    """Factory function to create the appropriate hotkey handler."""
    return WaylandHotkeyHandler()


if __name__ == "__main__":
    # Test the hotkey handler
    logging.basicConfig(level=logging.INFO)
    
    def test_callback():
        print("Hotkey triggered!")
    
    handler = create_wayland_hotkey_handler()
    if handler.setup_hotkey(test_callback):
        print("Hotkey handler setup successful. Press SUPER+A to test...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            handler.cleanup()
    else:
        print("Failed to setup hotkey handler")