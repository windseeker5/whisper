#!/home/kdresdell/DEV/whisper/venv/bin/python
"""
Test script for Wayland hotkey functionality.

This script helps verify that the hotkey system is working correctly
before running the full voice transcriber application.
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

# Add the bin directory to the path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from wayland_hotkey_handler import create_wayland_hotkey_handler
except ImportError as e:
    print(f"Error importing wayland_hotkey_handler: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HotkeyTester:
    """Test harness for hotkey functionality."""
    
    def __init__(self):
        self.trigger_count = 0
        self.handler = None
        self.running = True
    
    def test_callback(self):
        """Callback function for hotkey testing."""
        self.trigger_count += 1
        print(f"🎯 Hotkey triggered! (Count: {self.trigger_count})")
        logging.info(f"Hotkey callback executed - trigger count: {self.trigger_count}")
    
    def signal_handler(self, signum, frame):
        """Handle SIGUSR1 signal for testing."""
        print("📡 Received SIGUSR1 signal")
        self.test_callback()
    
    def run_test(self):
        """Run the hotkey test."""
        print("🧪 Whisper Hotkey Test Utility")
        print("=" * 40)
        
        # Setup signal handler
        signal.signal(signal.SIGUSR1, self.signal_handler)
        print("✅ Signal handler registered (SIGUSR1)")
        
        # Detect environment
        is_wayland = self._is_wayland_session()
        print(f"🖥️  Session type: {'Wayland' if is_wayland else 'X11'}")
        
        if is_wayland:
            print("🔧 Setting up Wayland-compatible hotkey handler...")
            self.handler = create_wayland_hotkey_handler()
            success = self.handler.setup_hotkey(self.test_callback)
            
            if success:
                print("✅ Wayland hotkey handler setup successful")
            else:
                print("❌ Wayland hotkey handler setup failed")
        else:
            print("ℹ️  X11 detected - pynput should work normally")
        
        # Display test information
        print("\n🎮 Test Methods:")
        print("1. Press SUPER+A (if Hyprland keybind is configured)")
        print("2. Send signal: pkill -SIGUSR1 -f test_hotkey.py")
        print("3. Use named pipe: echo 'trigger' > /tmp/whisper_hotkey_trigger")
        print("4. Press Ctrl+C to exit")
        print("\n⏳ Waiting for hotkey activation...")
        
        # Monitor named pipe if it exists
        pipe_path = "/tmp/whisper_hotkey_trigger"
        if os.path.exists(pipe_path):
            print(f"📡 Monitoring named pipe: {pipe_path}")
            import threading
            pipe_thread = threading.Thread(target=self._monitor_pipe, args=(pipe_path,), daemon=True)
            pipe_thread.start()
        
        # Main test loop
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        finally:
            self.cleanup()
    
    def _monitor_pipe(self, pipe_path):
        """Monitor the named pipe for triggers."""
        while self.running:
            try:
                with open(pipe_path, 'r') as pipe:
                    data = pipe.read().strip()
                    if data == 'trigger':
                        print("📦 Named pipe trigger received")
                        self.test_callback()
            except Exception as e:
                logging.debug(f"Pipe monitoring error: {e}")
                time.sleep(1)
    
    def _is_wayland_session(self):
        """Detect if running under Wayland."""
        return (os.environ.get('WAYLAND_DISPLAY') is not None or 
                os.environ.get('XDG_SESSION_TYPE') == 'wayland' or
                os.environ.get('HYPRLAND_INSTANCE_SIGNATURE') is not None)
    
    def cleanup(self):
        """Clean up test resources."""
        self.running = False
        if self.handler:
            self.handler.cleanup()
        print(f"\n📊 Test Summary:")
        print(f"   Total triggers: {self.trigger_count}")
        print(f"   Status: {'✅ Working' if self.trigger_count > 0 else '❌ No triggers detected'}")


def print_environment_info():
    """Print useful environment information for debugging."""
    print("\n🔍 Environment Information:")
    print("-" * 30)
    
    env_vars = [
        'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'HYPRLAND_INSTANCE_SIGNATURE',
        'DISPLAY', 'XDG_CURRENT_DESKTOP', 'DESKTOP_SESSION'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # Check for Hyprland
    try:
        import subprocess
        result = subprocess.run(['hyprctl', 'version'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"   Hyprland: ✅ Running")
        else:
            print(f"   Hyprland: ❌ Not responding")
    except:
        print(f"   Hyprland: ❌ Not available")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--env':
        print_environment_info()
        return
    
    tester = HotkeyTester()
    tester.run_test()


if __name__ == "__main__":
    main()