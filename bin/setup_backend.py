#!/usr/bin/env python3
"""
Backend Setup and Installation Tool

Helps users choose and configure their transcription backend (Whisper or Vosk).
Provides hardware recommendations and automatic model downloads.

Author: Python DevOps Automation Specialist
Compatible: Arch Linux, Wayland/Sway
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request
import zipfile
import shutil


class BackendSetup:
    """Setup tool for transcription backends."""

    VOSK_MODELS = {
        'en-small': {
            'name': 'vosk-model-small-en-us-0.15',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
            'size': '40 MB',
            'description': 'English (US) - Small, fast, good for old hardware'
        },
        'en-large': {
            'name': 'vosk-model-en-us-0.22',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
            'size': '1.8 GB',
            'description': 'English (US) - Large, accurate, requires more RAM'
        },
        'fr-small': {
            'name': 'vosk-model-small-fr-0.22',
            'url': 'https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip',
            'size': '41 MB',
            'description': 'French - Small, fast'
        }
    }

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "whisper_config.json"
        self.models_dir = Path("models")
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load existing configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✓ Configuration saved to {self.config_file}")

    def detect_system_specs(self) -> Dict[str, Any]:
        """Detect system specifications."""
        specs = {}

        try:
            import psutil
            specs['ram_gb'] = psutil.virtual_memory().total / (1024**3)
            specs['cpu_count'] = psutil.cpu_count()
        except ImportError:
            print("⚠ psutil not available, using defaults")
            specs['ram_gb'] = 4.0
            specs['cpu_count'] = 2

        return specs

    def recommend_backend(self, specs: Dict[str, Any]) -> str:
        """Recommend backend based on system specs.

        Args:
            specs: System specifications

        Returns:
            str: Recommended backend name
        """
        ram_gb = specs.get('ram_gb', 4.0)

        if ram_gb < 6:
            return 'vosk'
        else:
            return 'whisper'

    def interactive_setup(self) -> None:
        """Interactive backend setup."""
        print("\n" + "="*60)
        print("  Voice Transcription Backend Setup")
        print("="*60 + "\n")

        # Detect system
        specs = self.detect_system_specs()
        print(f"📊 System Information:")
        print(f"   RAM: {specs['ram_gb']:.1f} GB")
        print(f"   CPU Cores: {specs['cpu_count']}")
        print()

        # Recommend backend
        recommended = self.recommend_backend(specs)
        print(f"💡 Recommended backend: {recommended.upper()}")
        print()

        # Show backend comparison
        print("Backend Comparison:")
        print("-" * 60)
        print("Vosk:")
        print("  ✓ Very fast (10-20x faster than Whisper on old hardware)")
        print("  ✓ Low RAM usage (~50-200 MB)")
        print("  ✓ Works offline")
        print("  ✓ Perfect for 2007 iMac with 4GB RAM")
        print("  ✗ Lower accuracy than Whisper")
        print()
        print("Whisper:")
        print("  ✓ High accuracy")
        print("  ✓ Better language support")
        print("  ✓ More robust")
        print("  ✗ Slow on old hardware (30-60s for 10s audio)")
        print("  ✗ High RAM usage (500MB-2GB)")
        print("  ✗ Requires 6GB+ RAM for comfortable use")
        print("-" * 60)
        print()

        # Get user choice
        while True:
            choice = input(f"Choose backend (vosk/whisper) [default: {recommended}]: ").strip().lower()
            if not choice:
                choice = recommended

            if choice in ['vosk', 'whisper']:
                break
            print("❌ Invalid choice. Please enter 'vosk' or 'whisper'")

        print(f"\n✓ Selected backend: {choice.upper()}\n")

        # Setup chosen backend
        if choice == 'vosk':
            self.setup_vosk()
        else:
            self.setup_whisper()

        # Update config
        self.config['backend'] = choice
        self.save_config()

        print("\n" + "="*60)
        print("  Setup Complete!")
        print("="*60)
        print(f"\nYou can now run the application:")
        print(f"  python bin/voice_transcriber.py --tui")
        print()

    def setup_vosk(self) -> None:
        """Setup Vosk backend."""
        print("Setting up Vosk backend...")
        print()

        # Check if Vosk is installed
        try:
            import vosk
            print("✓ Vosk library is installed")
        except ImportError:
            print("❌ Vosk library not found")
            print("Install with: pip install vosk")
            return

        # Choose model
        print("\nAvailable Vosk models:")
        for i, (key, info) in enumerate(self.VOSK_MODELS.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   {info['description']}")
            print(f"   Size: {info['size']}")
            print()

        while True:
            choice = input("Select model number [1]: ").strip()
            if not choice:
                choice = "1"

            try:
                model_idx = int(choice) - 1
                model_key = list(self.VOSK_MODELS.keys())[model_idx]
                model_info = self.VOSK_MODELS[model_key]
                break
            except (ValueError, IndexError):
                print("❌ Invalid choice")

        # Download model
        model_path = self.models_dir / model_info['name']

        if model_path.exists():
            print(f"\n✓ Model already exists at {model_path}")
        else:
            print(f"\n📥 Downloading {model_info['name']}...")
            print(f"   Size: {model_info['size']}")
            self.download_vosk_model(model_info['url'], model_info['name'])

        # Update config
        self.config['vosk_model_path'] = str(model_path)
        print(f"✓ Vosk model configured: {model_path}")

    def setup_whisper(self) -> None:
        """Setup Whisper backend."""
        print("Setting up Whisper backend...")
        print()

        # Check if Whisper is installed
        try:
            import whisper
            print("✓ Whisper library is installed")
        except ImportError:
            print("❌ Whisper library not found")
            print("Install with: pip install openai-whisper")
            return

        # Choose model size
        models = {
            '1': ('tiny', '39 MB', 'Very fast, low accuracy'),
            '2': ('base', '74 MB', 'Good balance (RECOMMENDED)'),
            '3': ('small', '244 MB', 'Better accuracy, slower'),
            '4': ('medium', '769 MB', 'High accuracy, very slow on old hardware'),
            '5': ('large', '1550 MB', 'Best accuracy, extremely slow')
        }

        print("\nWhisper models:")
        for key, (name, size, desc) in models.items():
            print(f"{key}. {name} - {size} - {desc}")

        while True:
            choice = input("\nSelect model [2 for base]: ").strip()
            if not choice:
                choice = "2"

            if choice in models:
                model_name = models[choice][0]
                break
            print("❌ Invalid choice")

        # Update config
        self.config['whisper_model'] = model_name
        print(f"\n✓ Whisper model configured: {model_name}")
        print("   (Model will be downloaded on first use)")

    def download_vosk_model(self, url: str, model_name: str) -> None:
        """Download and extract Vosk model.

        Args:
            url: Download URL
            model_name: Model directory name
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.models_dir / f"{model_name}.zip"

        try:
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r   Progress: {percent:.1f}%", end='', flush=True)

            print(f"   Downloading from {url}")
            urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
            print("\n   ✓ Download complete")

            # Extract
            print("   Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)

            # Cleanup
            zip_path.unlink()
            print(f"   ✓ Model extracted to {self.models_dir / model_name}")

        except Exception as e:
            print(f"\n   ❌ Download failed: {e}")
            print(f"   Manual download: {url}")
            print(f"   Extract to: {self.models_dir}")

    def list_backends(self) -> None:
        """List available backends and their status."""
        print("\nAvailable Backends:")
        print("-" * 60)

        # Check Vosk
        try:
            import vosk
            vosk_status = "✓ Installed"
        except ImportError:
            vosk_status = "✗ Not installed (pip install vosk)"

        print(f"Vosk: {vosk_status}")

        # Check Whisper
        try:
            import whisper
            whisper_status = "✓ Installed"
        except ImportError:
            whisper_status = "✗ Not installed (pip install openai-whisper)"

        print(f"Whisper: {whisper_status}")
        print()

        # Show current configuration
        current = self.config.get('backend', 'not configured')
        print(f"Current backend: {current}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup transcription backend for voice-to-text application"
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive setup wizard'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available backends'
    )

    parser.add_argument(
        '--backend',
        choices=['vosk', 'whisper'],
        help='Set backend directly without interactive prompts'
    )

    args = parser.parse_args()

    setup = BackendSetup()

    if args.list:
        setup.list_backends()
    elif args.backend:
        setup.config['backend'] = args.backend
        print(f"Backend set to: {args.backend}")
        if args.backend == 'vosk':
            setup.setup_vosk()
        else:
            setup.setup_whisper()
        setup.save_config()
    else:
        # Default: interactive setup
        setup.interactive_setup()


if __name__ == "__main__":
    main()
