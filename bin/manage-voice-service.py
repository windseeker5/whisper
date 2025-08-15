#!/home/kdresdell/Documents/DEV/whisper/venv/bin/python
"""
Voice Transcriber Service Management Script

A comprehensive management tool for the voice transcriber systemd user service.
Provides commands to install, enable, start, stop, restart, and monitor the service.

Usage:
    python manage-voice-service.py install    # Install and enable the service
    python manage-voice-service.py start      # Start the service
    python manage-voice-service.py stop       # Stop the service
    python manage-voice-service.py restart    # Restart the service
    python manage-voice-service.py status     # Show service status
    python manage-voice-service.py logs       # Show recent logs
    python manage-voice-service.py uninstall  # Disable and remove the service

Author: Python DevOps Automation Specialist
Compatible: Arch Linux, Hyprland, systemd
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class VoiceServiceManager:
    """Manages the voice transcriber systemd user service."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.project_dir = self.script_dir.parent
        self.service_name = "voice-transcriber.service"
        self.service_file = self.project_dir / self.service_name
        self.systemd_user_dir = Path.home() / ".config" / "systemd" / "user"
        self.installed_service = self.systemd_user_dir / self.service_name
        
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a system command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"Error running command {' '.join(cmd)}: {e}")
            return subprocess.CompletedProcess(cmd, 1, "", str(e))
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print("Checking dependencies...")
        
        # Check systemctl
        result = self.run_command(["which", "systemctl"])
        if result.returncode != 0:
            print("ERROR: systemctl not found. systemd is required.")
            return False
        
        # Check if service files exist
        if not self.service_file.exists():
            print(f"ERROR: Service file not found: {self.service_file}")
            return False
            
        # Check wrapper script
        wrapper_script = self.script_dir / "voice-transcriber-service.sh"
        if not wrapper_script.exists():
            print(f"ERROR: Wrapper script not found: {wrapper_script}")
            return False
            
        # Check virtual environment
        venv_python = self.project_dir / "venv" / "bin" / "python"
        if not venv_python.exists():
            print(f"ERROR: Virtual environment not found: {venv_python}")
            return False
            
        print("✓ All dependencies found")
        return True
    
    def install_service(self) -> bool:
        """Install and enable the systemd user service."""
        print("Installing voice transcriber service...")
        
        if not self.check_dependencies():
            return False
        
        # Create systemd user directory
        self.systemd_user_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy service file
        try:
            shutil.copy2(self.service_file, self.installed_service)
            print(f"✓ Service file copied to {self.installed_service}")
        except Exception as e:
            print(f"ERROR: Failed to copy service file: {e}")
            return False
        
        # Reload systemd daemon
        result = self.run_command(["systemctl", "--user", "daemon-reload"])
        if result.returncode != 0:
            print(f"ERROR: Failed to reload systemd daemon: {result.stderr}")
            return False
        print("✓ Systemd daemon reloaded")
        
        # Enable service
        result = self.run_command(["systemctl", "--user", "enable", self.service_name])
        if result.returncode != 0:
            print(f"ERROR: Failed to enable service: {result.stderr}")
            return False
        print("✓ Service enabled for auto-start")
        
        print("\n🎉 Voice transcriber service installed successfully!")
        print("The service will start automatically on login.")
        print("Use 'python manage-voice-service.py start' to start it now.")
        return True
    
    def start_service(self) -> bool:
        """Start the voice transcriber service."""
        print("Starting voice transcriber service...")
        
        result = self.run_command(["systemctl", "--user", "start", self.service_name])
        if result.returncode != 0:
            print(f"ERROR: Failed to start service: {result.stderr}")
            return False
        
        # Wait a moment and check status
        time.sleep(2)
        if self.is_service_active():
            print("✓ Voice transcriber service started successfully")
            return True
        else:
            print("WARNING: Service started but may not be running properly")
            self.show_status()
            return False
    
    def stop_service(self) -> bool:
        """Stop the voice transcriber service."""
        print("Stopping voice transcriber service...")
        
        result = self.run_command(["systemctl", "--user", "stop", self.service_name])
        if result.returncode != 0:
            print(f"ERROR: Failed to stop service: {result.stderr}")
            return False
        
        print("✓ Voice transcriber service stopped")
        return True
    
    def restart_service(self) -> bool:
        """Restart the voice transcriber service."""
        print("Restarting voice transcriber service...")
        
        result = self.run_command(["systemctl", "--user", "restart", self.service_name])
        if result.returncode != 0:
            print(f"ERROR: Failed to restart service: {result.stderr}")
            return False
        
        # Wait a moment and check status
        time.sleep(2)
        if self.is_service_active():
            print("✓ Voice transcriber service restarted successfully")
            return True
        else:
            print("WARNING: Service restarted but may not be running properly")
            self.show_status()
            return False
    
    def is_service_active(self) -> bool:
        """Check if the service is currently active."""
        result = self.run_command(["systemctl", "--user", "is-active", self.service_name])
        return result.returncode == 0 and result.stdout.strip() == "active"
    
    def show_status(self) -> None:
        """Show detailed service status."""
        print(f"\n=== {self.service_name} Status ===")
        
        # Service status
        result = self.run_command(["systemctl", "--user", "status", self.service_name], capture_output=False)
        
        # Additional info
        print(f"\nService file: {self.installed_service}")
        print(f"Project directory: {self.project_dir}")
        print(f"Log directory: {self.project_dir}/log")
    
    def show_logs(self, lines: int = 50) -> None:
        """Show recent service logs."""
        print(f"\n=== Recent Logs (last {lines} lines) ===")
        
        # Show systemd journal logs
        result = self.run_command([
            "journalctl", "--user", "-u", self.service_name, 
            "-n", str(lines), "--no-pager"
        ], capture_output=False)
        
        # Also show application logs if available
        log_file = self.project_dir / "log" / "service.log"
        if log_file.exists():
            print(f"\n=== Service Log File ===")
            result = self.run_command(["tail", "-n", str(lines), str(log_file)], capture_output=False)
    
    def uninstall_service(self) -> bool:
        """Disable and remove the service."""
        print("Uninstalling voice transcriber service...")
        
        # Stop service if running
        if self.is_service_active():
            self.stop_service()
        
        # Disable service
        result = self.run_command(["systemctl", "--user", "disable", self.service_name])
        if result.returncode != 0:
            print(f"WARNING: Failed to disable service: {result.stderr}")
        else:
            print("✓ Service disabled")
        
        # Remove service file
        try:
            if self.installed_service.exists():
                self.installed_service.unlink()
                print(f"✓ Service file removed: {self.installed_service}")
        except Exception as e:
            print(f"ERROR: Failed to remove service file: {e}")
            return False
        
        # Reload daemon
        result = self.run_command(["systemctl", "--user", "daemon-reload"])
        if result.returncode != 0:
            print(f"WARNING: Failed to reload systemd daemon: {result.stderr}")
        else:
            print("✓ Systemd daemon reloaded")
        
        print("✓ Voice transcriber service uninstalled")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage voice transcriber systemd user service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  install    Install and enable the service for auto-start
  start      Start the service
  stop       Stop the service
  restart    Restart the service
  status     Show detailed service status
  logs       Show recent service logs
  uninstall  Disable and remove the service

Examples:
  python manage-voice-service.py install
  python manage-voice-service.py start
  python manage-voice-service.py logs
        """
    )
    
    parser.add_argument(
        "command",
        choices=["install", "start", "stop", "restart", "status", "logs", "uninstall"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--log-lines",
        type=int,
        default=50,
        help="Number of log lines to show (default: 50)"
    )
    
    args = parser.parse_args()
    
    manager = VoiceServiceManager()
    
    try:
        if args.command == "install":
            success = manager.install_service()
        elif args.command == "start":
            success = manager.start_service()
        elif args.command == "stop":
            success = manager.stop_service()
        elif args.command == "restart":
            success = manager.restart_service()
        elif args.command == "status":
            manager.show_status()
            success = True
        elif args.command == "logs":
            manager.show_logs(args.log_lines)
            success = True
        elif args.command == "uninstall":
            success = manager.uninstall_service()
        else:
            print(f"Unknown command: {args.command}")
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()