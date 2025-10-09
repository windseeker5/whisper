#!/home/kdresdell/Documents/DEV/whisper/venv/bin/python
"""
Voice-to-Text Application using OpenAI Whisper

A comprehensive voice transcription tool for Arch Linux with Hyprland (Wayland) that
provides global hotkey activation, audio processing, and desktop integration.

Features:
- Global hotkey (SUPER+A) for recording control
- Voice Activity Detection and noise reduction
- OpenAI Whisper transcription
- Clipboard integration and desktop notifications
- Configurable microphone and model selection
- Timestamped logging

Author: Python DevOps Automation Specialist
Compatible: Arch Linux, Hyprland, Wayland
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Simple audio system detection (inline implementation)
from enum import Enum

class AudioSystem(Enum):
    PULSEAUDIO = "pulseaudio"
    PIPEWIRE = "pipewire"
    ALSA = "alsa"
    UNKNOWN = "unknown"

class AudioDevice:
    def __init__(self, name: str, device_id: str, system: AudioSystem, channels: int = 1, 
                 sample_rate: int = 44100, is_default: bool = False):
        self.name = name
        self.device_id = device_id
        self.system = system
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_default = is_default

class AudioSystemInfo:
    def __init__(self):
        self.primary_system = AudioSystem.UNKNOWN
        self.pulseaudio_compatible = False
        self.alsa_compatible = False
        self.input_devices = []
        self.default_device = None

class SimpleAudioDetector:
    """Simplified audio system detector."""
    
    def detect_audio_system(self) -> AudioSystemInfo:
        """Detect the current audio system."""
        info = AudioSystemInfo()
        
        # Simple detection logic
        try:
            import subprocess
            # Check for PulseAudio/PipeWire
            result = subprocess.run(['pulseaudio', '--check'], capture_output=True)
            if result.returncode == 0:
                info.primary_system = AudioSystem.PULSEAUDIO
                info.pulseaudio_compatible = True
            else:
                # Check for PipeWire
                result = subprocess.run(['pipewire', '--version'], capture_output=True)
                if result.returncode == 0:
                    info.primary_system = AudioSystem.PIPEWIRE
                    info.pulseaudio_compatible = True
                else:
                    info.primary_system = AudioSystem.ALSA
                    info.alsa_compatible = True
        except:
            info.primary_system = AudioSystem.ALSA
            info.alsa_compatible = True
        
        return info
    
    def get_recommended_audio_backend(self, audio_info: AudioSystemInfo) -> str:
        """Get recommended audio backend."""
        if audio_info.pulseaudio_compatible:
            return 'pulse'
        return 'alsa'

try:
    # Suppress PyAudio error messages at the C level
    import os
    import sys
    
    # Set environment variables to reduce ALSA/PyAudio verbosity
    os.environ['ALSA_CAPTURE_DELAY'] = '0'
    os.environ['ALSA_PLAYBACK_DELAY'] = '0' 
    os.environ['ALSA_MIXER_ELEMENTS'] = '0'
    
    # Temporarily redirect stderr during PyAudio import
    stderr_backup = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        import pyaudio
    finally:
        sys.stderr.close()
        sys.stderr = stderr_backup

    # Import whisper optionally (only needed for direct WhisperTranscriber, not backend abstraction)
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False
        whisper = None  # Set to None so code doesn't crash on reference

    import numpy as np

    # Import pynput optionally (only needed for hotkey mode, not TUI)
    try:
        from pynput import keyboard
        PYNPUT_AVAILABLE = True
    except Exception:  # Catch any error (ImportError, DisplayNameError, etc.)
        PYNPUT_AVAILABLE = False
        keyboard = None

    import pyperclip
    import sounddevice as sd
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    
    # Optional dependencies for enhanced audio processing
    try:
        import scipy.signal
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        logging.warning("SciPy not available. Some advanced audio processing features will be disabled.")
    
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except ImportError:
        LIBROSA_AVAILABLE = False
        logging.debug("Librosa not available. Advanced audio analysis features will be disabled.")

except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install core dependencies with: pip install -r requirements-core.txt")
    print("For Whisper backend (optional): ./install-whisper.sh")
    sys.exit(1)


class WhisperConfig:
    """Configuration management for the voice-to-text application."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "whisper_config.json"
        self.config_dir.mkdir(exist_ok=True)
        self.default_config = {
            "microphone_device": None,
            "whisper_model": "base",
            "language": "auto",
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "vad_threshold": 0.01,
            "silence_threshold": -30,
            "min_silence_len": 500,
            "notification_enabled": True,
            "sound_feedback": True,
            "sound_feedback_verbose": True,  # Show sound feedback messages in logs
            # Audio level and gain settings
            "microphone_gain": 1.0,  # Software gain multiplier (1.0 = no gain, 2.0 = double volume)
            "auto_gain_control": True,  # Enable automatic gain control
            "target_rms_level": 0.1,  # Target RMS level for AGC (0.0-1.0)
            "gain_boost_db": 0,  # Additional gain boost in decibels
            "system_mic_boost": False,  # Enable system-level microphone boost
            "show_audio_levels": True,  # Show real-time audio levels
            "level_update_interval": 0.1,  # Update interval for level display (seconds)
            "manual_mode_levels": "minimal",  # Level display in manual mode: 'full', 'minimal', 'off'
            "noise_gate_threshold": -50,  # Noise gate threshold in dB
            "compressor_enabled": True,  # Enable audio compression
            "normalize_audio": True  # Normalize audio levels before transcription
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults to handle new config options
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.warning(f"Error loading config: {e}. Using defaults.")
        return self.default_config.copy()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value) -> None:
        """Set configuration value."""
        self.config[key] = value


class AudioProcessor:
    """Audio recording and processing functionality with automatic audio system detection."""
    
    # Common sample rates to test (in Hz) - ordered by preference for speech
    PREFERRED_SAMPLE_RATES = [16000, 44100, 48000, 22050, 32000, 8000, 11025]
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.validated_config = None
        
        # Audio level monitoring
        self.current_audio_level = 0.0
        self.peak_audio_level = 0.0
        self.level_history = []
        self.level_lock = threading.Lock()
        self.level_monitor_thread = None
        self.show_levels = config.get('show_audio_levels', True)
        
        # Gain control
        self.current_gain = config.get('microphone_gain', 1.0)
        
        # Manual mode tracking
        self.manual_mode = False
        self.auto_gain_enabled = config.get('auto_gain_control', True)
        self.target_rms = config.get('target_rms_level', 0.1)
        self.gain_boost_db = config.get('gain_boost_db', 0)
        
        # Audio processing
        self.normalize_audio = config.get('normalize_audio', True)
        self.compressor_enabled = config.get('compressor_enabled', True)
        self.noise_gate_threshold = config.get('noise_gate_threshold', -50)
        
        # Detect audio system automatically
        self.audio_detector = SimpleAudioDetector()
        self.audio_system_info = self.audio_detector.detect_audio_system()
        
        # Initialize PyAudio with appropriate backend
        self._initialize_audio_backend()
        
        # Log detected audio system
        logging.info(f"Detected audio system: {self.audio_system_info.primary_system.value}")
        logging.info(f"Recommended backend: {self.audio_detector.get_recommended_audio_backend(self.audio_system_info)}")
        
        # Initialize PyAudio with comprehensive error suppression
        import contextlib
        import os
        import sys
        
        # Save original stderr
        original_stderr = sys.stderr
        
        try:
            # Redirect stderr to suppress PyAudio initialization errors
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                self.audio = pyaudio.PyAudio()
        finally:
            # Restore original stderr
            sys.stderr = original_stderr
        
        # Validate and auto-fix audio configuration on initialization
        self._validate_audio_config()
        
        # Apply system-level microphone boost if enabled
        if self.config.get('system_mic_boost', False):
            self.apply_system_mic_boost()
    
    def _initialize_audio_backend(self) -> None:
        """Initialize audio backend based on detected system."""
        backend = self.audio_detector.get_recommended_audio_backend(self.audio_system_info)
        
        # Set environment variables for optimal compatibility
        if backend == 'pulse':
            # Ensure PulseAudio/PipeWire compatibility
            os.environ.setdefault('PULSE_RUNTIME_PATH', 
                                f'/run/user/{os.getuid()}/pulse')
        elif backend == 'alsa':
            # Configure ALSA if needed
            pass
        
        logging.info(f"Audio backend configured: {backend}")
    
    def _validate_audio_config(self) -> None:
        """Validate current audio configuration and auto-fix if necessary."""
        device_id = self.config.get('microphone_device')
        sample_rate = self.config.get('sample_rate', 16000)
        channels = self.config.get('channels', 1)
        
        logging.info("Validating audio configuration...")
        
        # Get device index for testing
        device_index = self._get_pyaudio_device_index(device_id)
        
        # Test current configuration with error suppression
        if self._test_audio_config(device_index, sample_rate, channels):
            logging.info("✓ Current audio configuration is valid")
            self.validated_config = {
                'device_index': device_index,
                'sample_rate': sample_rate,
                'channels': channels
            }
            return
        
        logging.warning("✗ Current audio configuration is not compatible")
        
        # Try to find a working configuration
        working_config = self._find_working_config(device_index)
        
        if working_config:
            self.validated_config = working_config
            # Update the main config with working values
            self.config.set('sample_rate', working_config['sample_rate'])
            self.config.set('channels', working_config['channels'])
            self.config.save_config()
            
            logging.info(f"✓ Auto-configured compatible settings:")
            logging.info(f"  Sample Rate: {working_config['sample_rate']} Hz")
            logging.info(f"  Channels: {working_config['channels']}")
        else:
            logging.error("✗ Could not find any working audio configuration")
            logging.error("Please run audio diagnostics: python bin/audio_diagnostics.py")
    
    def _test_audio_config(self, device_index: Optional[int], sample_rate: int, 
                          channels: int) -> bool:
        """Test if a specific audio configuration works."""
        try:
            import subprocess
            import sys
            import os
            import contextlib
            
            # For C-level error suppression, we need to redirect stderr at the file descriptor level
            original_stderr_fd = os.dup(2)  # Duplicate stderr file descriptor
            
            try:
                # Redirect stderr file descriptor to devnull
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 2)  # Redirect stderr
                
                # Try to create a stream with the given parameters
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=1024,
                    start=False  # Don't start the stream, just test creation
                )
                stream.close()
                return True
                
            finally:
                # Always restore stderr file descriptor
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
                if 'devnull_fd' in locals():
                    os.close(devnull_fd)
                
        except Exception:
            # Silently fail during testing
            return False
    
    def _find_working_config(self, device_index: Optional[int]) -> Optional[Dict[str, Any]]:
        """Find a working audio configuration for the device with efficient testing."""
        logging.info("Searching for compatible audio configuration...")
        
        # If device_index is None, try to find the default device
        if device_index is None:
            try:
                default_device = self.audio.get_default_input_device_info()
                device_index = default_device['index']
                logging.info(f"Using default input device: {default_device['name']}")
            except Exception as e:
                logging.warning(f"Could not get default device: {e}")
                # Try device index 0 as fallback
                device_index = 0
        
        # Quick test with most common configurations first
        quick_test_configs = [
            (16000, 1),   # 16kHz mono (best for speech)
            (44100, 1),   # CD quality mono
            (48000, 1),   # Professional mono
            (16000, 2),   # 16kHz stereo
            (44100, 2),   # CD quality stereo
        ]
        
        for sample_rate, channels in quick_test_configs:
            if self._test_audio_config(device_index, sample_rate, channels):
                logging.info(f"✓ Found working config: {sample_rate} Hz, {channels} channels")
                return {
                    'device_index': device_index,
                    'sample_rate': sample_rate,
                    'channels': channels
                }
        
        # If quick test fails, try alternative devices but limit the search
        logging.info("Trying alternative devices...")
        device_count = min(self.audio.get_device_count(), 10)  # Limit to first 10 devices
        
        for alt_device_idx in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(alt_device_idx)
                if device_info['maxInputChannels'] > 0:  # Only input devices
                    # Test only the most basic configuration
                    if self._test_audio_config(alt_device_idx, 16000, 1):
                        logging.info(f"✓ Found working device {alt_device_idx}: {device_info['name']}")
                        logging.info(f"  Config: 16000 Hz, 1 channel")
                        
                        # Update device in config too
                        self.config.set('microphone_device', f"pyaudio:{alt_device_idx}")
                        
                        return {
                            'device_index': alt_device_idx,
                            'sample_rate': 16000,
                            'channels': 1
                        }
            except Exception:
                continue  # Silently skip problematic devices
        
        return None
    
    def get_validated_config(self) -> Optional[Dict[str, Any]]:
        """Get the validated audio configuration."""
        return self.validated_config
    
    def get_audio_system_info(self) -> AudioSystemInfo:
        """Get detected audio system information."""
        return self.audio_system_info
        
    def get_audio_devices(self, suppress_errors: bool = False) -> List[Dict]:
        """Get list of available audio input devices using detected system info."""
        devices = []
        
        # Use our detected audio devices as primary source
        for audio_device in self.audio_system_info.input_devices:
            devices.append({
                'name': audio_device.name,
                'device_id': audio_device.device_id,
                'system': audio_device.system.value,
                'channels': audio_device.channels,
                'sample_rate': audio_device.sample_rate,
                'is_default': audio_device.is_default
            })
        
        # Also get PyAudio devices for compatibility
        try:
            import contextlib
            
            # Optionally suppress PyAudio errors
            context_manager = contextlib.redirect_stderr(open(os.devnull, 'w')) if suppress_errors else contextlib.nullcontext()
            
            with context_manager:
                for i in range(self.audio.get_device_count()):
                    try:
                        device_info = self.audio.get_device_info_by_index(i)
                        if device_info['maxInputChannels'] > 0:
                            # Check if this device is already in our list
                            device_name = device_info['name']
                            if not any(d['name'] == device_name for d in devices):
                                devices.append({
                                    'index': i,
                                    'name': device_name,
                                    'device_id': f"pyaudio:{i}",
                                    'system': 'pyaudio',
                                    'channels': device_info['maxInputChannels'],
                                    'sample_rate': int(device_info['defaultSampleRate']),
                                    'is_default': False
                                })
                    except Exception:
                        if not suppress_errors:
                            logging.debug(f"Could not get info for device {i}")
                        continue
        except Exception as e:
            if not suppress_errors:
                logging.warning(f"Error getting PyAudio devices: {e}")
        
        return devices
    
    def _get_pyaudio_device_index(self, device_id) -> Optional[int]:
        """Convert device_id to PyAudio device index."""
        if device_id is None:
            return None
        
        # If it's already a PyAudio index
        if isinstance(device_id, int):
            return device_id
        
        # If it's a PyAudio device string
        if isinstance(device_id, str) and device_id.startswith('pyaudio:'):
            try:
                return int(device_id.split(':')[1])
            except (ValueError, IndexError):
                logging.warning(f"Invalid PyAudio device ID: {device_id}")
                return None
        
        # Try to find matching device by name for PulseAudio/PipeWire devices
        if isinstance(device_id, str):
            try:
                for i in range(self.audio.get_device_count()):
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        # Try to match device names
                        if device_id in device_info['name'] or device_info['name'] in device_id:
                            logging.info(f"Matched device '{device_id}' to PyAudio index {i}")
                            return i
            except Exception as e:
                logging.warning(f"Error matching device: {e}")
        
        # Return None to use default device
        logging.warning(f"Could not match device '{device_id}', using default")
        return None
    
    def calculate_audio_level(self, audio_data: bytes) -> Tuple[float, float]:
        """Calculate RMS and peak audio levels from audio data.
        
        Returns:
            Tuple[float, float]: (rms_level, peak_level) in range 0.0-1.0
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0, 0.0
            
            # Normalize to float range -1.0 to 1.0
            normalized_audio = audio_array.astype(np.float32) / 32768.0
            
            # Calculate RMS (Root Mean Square) level
            rms_level = np.sqrt(np.mean(normalized_audio ** 2))
            
            # Calculate peak level
            peak_level = np.max(np.abs(normalized_audio))
            
            return float(rms_level), float(peak_level)
            
        except Exception as e:
            logging.debug(f"Error calculating audio level: {e}")
            return 0.0, 0.0
    
    def calculate_db_level(self, rms_level: float) -> float:
        """Convert RMS level to decibels.
        
        Args:
            rms_level: RMS level in range 0.0-1.0
            
        Returns:
            float: Level in decibels (dB)
        """
        if rms_level <= 0:
            return -80.0  # Very quiet threshold
        return 20 * np.log10(rms_level)
    
    def apply_gain_and_processing(self, audio_data: bytes) -> bytes:
        """Apply gain, AGC, and audio processing to audio data.
        
        Args:
            audio_data: Raw audio data as bytes
            
        Returns:
            bytes: Processed audio data
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            if len(audio_array) == 0:
                return audio_data
            
            # Normalize to -1.0 to 1.0 range
            audio_normalized = audio_array / 32768.0
            
            # Apply noise gate
            rms_level = np.sqrt(np.mean(audio_normalized ** 2))
            db_level = self.calculate_db_level(rms_level)
            
            if db_level < self.noise_gate_threshold:
                # Audio below noise gate threshold, attenuate
                audio_normalized *= 0.1
            
            # Apply automatic gain control
            if self.auto_gain_enabled and rms_level > 0:
                # Calculate AGC gain
                gain_adjustment = self.target_rms / rms_level
                # Limit gain adjustment to prevent extreme values
                gain_adjustment = np.clip(gain_adjustment, 0.1, 10.0)
                
                # Smooth gain changes to prevent artifacts
                self.current_gain = 0.9 * self.current_gain + 0.1 * gain_adjustment
            
            # Apply software gain
            total_gain = self.current_gain
            
            # Apply gain boost (convert dB to linear)
            if self.gain_boost_db != 0:
                boost_linear = 10 ** (self.gain_boost_db / 20.0)
                total_gain *= boost_linear
            
            # Apply gain
            audio_normalized *= total_gain
            
            # Apply compression if enabled
            if self.compressor_enabled:
                # Simple soft compression
                threshold = 0.7
                ratio = 3.0
                
                # Apply compression to values above threshold
                mask = np.abs(audio_normalized) > threshold
                compressed_values = np.sign(audio_normalized[mask]) * (
                    threshold + (np.abs(audio_normalized[mask]) - threshold) / ratio
                )
                audio_normalized[mask] = compressed_values
            
            # Clip to prevent distortion
            audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
            
            # Convert back to int16
            audio_processed = (audio_normalized * 32767).astype(np.int16)
            
            return audio_processed.tobytes()
            
        except Exception as e:
            logging.debug(f"Error applying audio processing: {e}")
            return audio_data
    
    def start_level_monitoring(self) -> None:
        """Start the audio level monitoring thread."""
        if self.level_monitor_thread and self.level_monitor_thread.is_alive():
            return
        
        def level_monitor():
            """Monitor and display audio levels in real-time."""
            last_display_time = 0
            
            while self.is_recording:
                try:
                    with self.level_lock:
                        current_level = self.current_audio_level
                        peak_level = self.peak_audio_level
                    
                    current_time = time.time()
                    
                    # Determine display settings based on mode
                    manual_mode_setting = self.config.get('manual_mode_levels', 'minimal')
                    should_display = False
                    update_interval = self.config.get('level_update_interval', 0.1)
                    
                    if self.show_levels and current_level > 0:
                        if self.manual_mode:
                            # Manual mode: different behavior based on setting
                            if manual_mode_setting == 'off':
                                should_display = False
                            elif manual_mode_setting == 'minimal':
                                # Update less frequently in manual mode (every 0.5 seconds)
                                should_display = (current_time - last_display_time) >= 0.5
                            elif manual_mode_setting == 'full':
                                # Same as normal mode
                                should_display = (current_time - last_display_time) >= update_interval
                        else:
                            # Normal mode: use configured interval
                            should_display = (current_time - last_display_time) >= update_interval
                    
                    if should_display:
                        # Convert to dB for display
                        db_level = self.calculate_db_level(current_level)
                        peak_db = self.calculate_db_level(peak_level)
                        
                        # Create level meter
                        level_bar = self.create_level_bar(current_level, peak_level)
                        
                        if self.manual_mode and manual_mode_setting == 'minimal':
                            # Minimal display for manual mode - shorter format
                            print(f"\rLevel: {level_bar} {db_level:+4.0f}dB", end="", flush=True)
                        else:
                            # Full display
                            print(f"\rAudio Level: {level_bar} {db_level:+5.1f}dB (Peak: {peak_db:+5.1f}dB) Gain: {self.current_gain:.2f}x", end="", flush=True)
                        
                        last_display_time = current_time
                    
                    # Sleep for a shorter interval to maintain responsiveness
                    time.sleep(0.05)
                    
                except Exception as e:
                    logging.debug(f"Error in level monitoring: {e}")
                    break
        
        self.level_monitor_thread = threading.Thread(target=level_monitor, daemon=True)
        self.level_monitor_thread.start()
    
    def create_level_bar(self, rms_level: float, peak_level: float, width: int = 20) -> str:
        """Create a visual level bar for audio levels.
        
        Args:
            rms_level: RMS audio level (0.0-1.0)
            peak_level: Peak audio level (0.0-1.0)
            width: Width of the level bar in characters
            
        Returns:
            str: Visual level bar
        """
        try:
            # Scale levels to bar width
            rms_bar_level = int(rms_level * width)
            peak_bar_level = int(peak_level * width)
            
            # Create the bar
            bar = []
            for i in range(width):
                if i < rms_bar_level:
                    if i < width * 0.6:  # Green zone
                        bar.append('█')
                    elif i < width * 0.8:  # Yellow zone
                        bar.append('▓')
                    else:  # Red zone
                        bar.append('▒')
                elif i == peak_bar_level and i >= rms_bar_level:
                    bar.append('|')  # Peak indicator
                else:
                    bar.append(' ')
            
            return '[' + ''.join(bar) + ']'
            
        except Exception:
            return '[' + ' ' * width + ']'
    
    def apply_system_mic_boost(self, boost_percentage: int = 50) -> bool:
        """Apply system-level microphone boost using PulseAudio/PipeWire commands.
        
        Args:
            boost_percentage: Boost percentage (0-150, where 100 = no change)
            
        Returns:
            bool: True if boost was applied successfully
        """
        try:
            # Get current microphone device
            device_id = self.config.get('microphone_device')
            if not device_id:
                logging.warning("No microphone device configured for system boost")
                return False
            
            # Try to get the actual PulseAudio/PipeWire source name
            source_name = self._get_system_source_name(device_id)
            if not source_name:
                logging.warning("Could not determine system source name for boost")
                return False
            
            # Calculate volume level (PulseAudio uses percentage)
            base_volume = 100  # Base volume percentage
            target_volume = base_volume + boost_percentage
            target_volume = max(0, min(target_volume, 200))  # Clamp to reasonable range
            
            # Apply the boost using pactl
            cmd = ['pactl', 'set-source-volume', source_name, f'{target_volume}%']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                logging.info(f"Applied system microphone boost: {target_volume}% on {source_name}")
                return True
            else:
                logging.warning(f"Failed to apply microphone boost: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("Timeout applying microphone boost")
            return False
        except Exception as e:
            logging.error(f"Error applying system microphone boost: {e}")
            return False
    
    def _get_system_source_name(self, device_id: str) -> Optional[str]:
        """Get the system source name for a device ID.
        
        Args:
            device_id: Device ID from configuration
            
        Returns:
            str: System source name or None if not found
        """
        try:
            # If it's a PyAudio device, try to map it to system source
            if isinstance(device_id, str) and device_id.startswith('pyaudio:'):
                device_index = int(device_id.split(':')[1])
                device_info = self.audio.get_device_info_by_index(device_index)
                device_name = device_info['name']
                
                # Try to find matching PulseAudio source
                return self._find_pulse_source_by_name(device_name)
            
            # If it's already a system source name, return it
            elif isinstance(device_id, str) and not device_id.startswith('pyaudio:'):
                return device_id
            
            # Try to get default source
            result = subprocess.run(['pactl', 'get-default-source'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return result.stdout.strip()
                
        except Exception as e:
            logging.debug(f"Error getting system source name: {e}")
        
        return None
    
    def _find_pulse_source_by_name(self, device_name: str) -> Optional[str]:
        """Find PulseAudio source by device name.
        
        Args:
            device_name: Device name to search for
            
        Returns:
            str: PulseAudio source name or None if not found
        """
        try:
            result = subprocess.run(['pactl', 'list', 'short', 'sources'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None
            
            # Parse the output to find matching source
            for line in result.stdout.split('\n'):
                if line.strip() and device_name.lower() in line.lower():
                    # Extract source name (first column)
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        return parts[1]
            
            # If no exact match, try partial matching
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        # Check if any part of the device name matches
                        device_words = device_name.lower().split()
                        source_words = source_name.lower().split('.')
                        
                        for device_word in device_words:
                            if len(device_word) > 3:  # Only check meaningful words
                                for source_word in source_words:
                                    if device_word in source_word or source_word in device_word:
                                        return source_name
                        
        except Exception as e:
            logging.debug(f"Error finding PulseAudio source: {e}")
        
        return None
    
    def get_current_mic_volume(self) -> Optional[float]:
        """Get current microphone volume level from system.
        
        Returns:
            float: Volume level as percentage (0-200) or None if unavailable
        """
        try:
            device_id = self.config.get('microphone_device')
            source_name = self._get_system_source_name(device_id)
            
            if not source_name:
                return None
            
            result = subprocess.run(['pactl', 'list', 'sources'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None
            
            # Parse output to find volume for our source
            in_target_source = False
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('Name:') and source_name in line:
                    in_target_source = True
                elif in_target_source and line.startswith('Volume:'):
                    # Extract volume percentage
                    # Format: "Volume: front-left: 32768 /  50% / -18.06 dB,   front-right: 32768 /  50% / -18.06 dB"
                    parts = line.split('/')
                    if len(parts) >= 2:
                        try:
                            volume_str = parts[1].strip().replace('%', '')
                            return float(volume_str)
                        except ValueError:
                            pass
                elif in_target_source and line.startswith('Source #'):
                    # Moved to next source
                    break
                    
        except Exception as e:
            logging.debug(f"Error getting microphone volume: {e}")
        
        return None
    
    def stop_level_monitoring(self) -> None:
        """Stop the audio level monitoring thread."""
        if self.level_monitor_thread and self.level_monitor_thread.is_alive():
            self.level_monitor_thread.join(timeout=1.0)
        
        # Clear the level display line
        if self.show_levels:
            # Clear the entire line and move cursor to beginning
            print("\r" + " " * 100 + "\r", end="", flush=True)
            # In manual mode, add a newline to separate from next prompt
            if self.manual_mode:
                print()
    
    def start_recording(self) -> bool:
        """Start audio recording with validated configuration."""
        if self.is_recording:
            return False
        
        # Check if we have a validated configuration
        if not self.validated_config:
            logging.error("No valid audio configuration available")
            return False
            
        try:
            # Use validated configuration parameters
            device_index = self.validated_config['device_index']
            sample_rate = self.validated_config['sample_rate']
            channels = self.validated_config['channels']
            chunk_size = self.config.get('chunk_size', 1024)
            
            logging.info(f"Starting recording with validated config:")
            logging.info(f"  Device: {device_index}")
            logging.info(f"  Sample Rate: {sample_rate} Hz")
            logging.info(f"  Channels: {channels}")
            
            def audio_callback(in_data, frame_count, time_info, status):
                if self.is_recording:
                    # Calculate audio levels for monitoring (use original data)
                    rms_level, peak_level = self.calculate_audio_level(in_data)
                    
                    # Update level monitoring
                    with self.level_lock:
                        self.current_audio_level = rms_level
                        self.peak_audio_level = max(self.peak_audio_level * 0.95, peak_level)  # Peak decay
                        
                        # Keep history for AGC
                        self.level_history.append(rms_level)
                        if len(self.level_history) > 100:  # Keep last 100 samples
                            self.level_history.pop(0)
                    
                    # Store the original audio data without real-time processing
                    # Audio processing will be done during file saving for better quality
                    self.audio_data.append(in_data)
                    
                return (None, pyaudio.paContinue)
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size,
                stream_callback=audio_callback
            )
            
            self.audio_data = []
            self.is_recording = True
            
            # Reset level monitoring
            with self.level_lock:
                self.current_audio_level = 0.0
                self.peak_audio_level = 0.0
                self.level_history.clear()
            
            # Start level monitoring if enabled
            if self.show_levels:
                self.start_level_monitoring()
            
            self.stream.start_stream()
            logging.info("✓ Recording started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            logging.error("Try running: python bin/audio_diagnostics.py --fix-config")
            return False
    
    def stop_recording(self) -> Optional[tuple[str, str]]:
        """Stop recording and save audio file.
        
        Returns:
            Tuple of (file_path, timestamp) if successful, None if failed.
            The timestamp is in format 'YYYY-mm-dd HH:MM:SS' for logging consistency.
        """
        if not self.is_recording:
            return None
            
        try:
            self.is_recording = False
            
            # Stop level monitoring
            self.stop_level_monitoring()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if not self.audio_data:
                logging.warning("No audio data recorded")
                return None
            
            # Combine audio data
            audio_bytes = b''.join(self.audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Save to temporary file using validated sample rate
            # Generate both filename timestamp and log timestamp from the same datetime
            now = datetime.now()
            filename_timestamp = now.strftime("%Y%m%d_%H_M%S")
            log_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            
            rec_dir = Path("rec")
            rec_dir.mkdir(exist_ok=True)
            audio_file = rec_dir / f"recording_{filename_timestamp}.wav"
            
            # Convert to float32 and normalize to prevent clipping
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Apply minimal gain if configured
            microphone_gain = self.config.get('microphone_gain', 1.0)
            if microphone_gain != 1.0:
                audio_float *= microphone_gain
                # Prevent clipping
                audio_float = np.clip(audio_float, -1.0, 1.0)
            
            # Use the validated sample rate, not the config one (in case they differ)
            sample_rate = self.validated_config['sample_rate'] if self.validated_config else self.config.get('sample_rate')
            sf.write(str(audio_file), audio_float, sample_rate)
            logging.info(f"Recording saved to {audio_file} (sample rate: {sample_rate} Hz, gain: {microphone_gain}x)")
            return (str(audio_file), log_timestamp)
            
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            return None
    
    def process_audio(self, audio_file: str) -> Optional[str]:
        """Process audio with minimal processing to maintain quality."""
        try:
            # For better quality, we'll do minimal processing
            # The main issue was real-time processing causing timing problems
            
            # Load audio
            audio = AudioSegment.from_wav(audio_file)
            
            # Only remove excessive silence (longer than 1 second)
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # 1 second minimum silence
                silence_thresh=self.config.get('silence_threshold', -30),
                keep_silence=200  # Keep 200ms of silence around speech
            )
            
            if not chunks:
                logging.warning("No audio content after silence removal")
                return audio_file  # Return original file if no chunks
            
            # Combine chunks with minimal processing
            processed_audio = AudioSegment.empty()
            for chunk in chunks:
                processed_audio += chunk
            
            # Save the minimally processed audio
            processed_file = audio_file.replace('.wav', '_processed.wav')
            processed_audio.export(processed_file, format="wav")
            
            logging.info("Minimal audio processing completed - preserving original quality")
            return processed_file
                
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return audio_file  # Return original file if processing fails
    
    def normalize_audio_segment(self, audio: AudioSegment) -> AudioSegment:
        """Normalize an audio segment to optimal levels.
        
        Args:
            audio: AudioSegment to normalize
            
        Returns:
            AudioSegment: Normalized audio
        """
        try:
            # Calculate current RMS and peak levels
            current_rms = audio.rms
            
            if current_rms == 0:
                return audio
            
            # Target RMS level for speech (roughly -20 dB)
            target_rms = 1000  # Adjust based on your needs
            
            # Calculate gain needed
            gain_needed = target_rms / current_rms
            
            # Limit gain to prevent extreme amplification
            gain_needed = min(gain_needed, 10.0)  # Max 10x gain (20 dB)
            
            # Apply gain
            if gain_needed > 1.1 or gain_needed < 0.9:  # Only adjust if significant difference
                gain_db = 20 * np.log10(gain_needed)
                audio = audio + gain_db
                logging.debug(f"Normalized audio: {gain_needed:.2f}x ({gain_db:+.1f} dB)")
            
            return audio
            
        except Exception as e:
            logging.debug(f"Error normalizing audio: {e}")
            return audio
    
    def calculate_optimal_volume_boost(self, audio: AudioSegment) -> float:
        """Calculate optimal volume boost for audio segment.
        
        Args:
            audio: AudioSegment to analyze
            
        Returns:
            float: Volume boost multiplier
        """
        try:
            # Get audio statistics
            rms_level = audio.rms
            max_level = audio.max
            
            if rms_level == 0:
                return 1.0
            
            # Target levels for good speech recognition
            target_rms = 2000   # Target RMS level
            max_safe_level = 20000  # Maximum safe level to prevent clipping
            
            # Calculate boost based on RMS
            rms_boost = target_rms / rms_level
            
            # Check if boost would cause clipping
            if max_level * rms_boost > max_safe_level:
                # Limit boost to prevent clipping
                rms_boost = max_safe_level / max_level * 0.9  # 90% of max to be safe
            
            # Reasonable limits
            rms_boost = max(0.5, min(rms_boost, 8.0))  # Between 0.5x and 8x
            
            return rms_boost
            
        except Exception as e:
            logging.debug(f"Error calculating volume boost: {e}")
            return 1.0
    
    def validate_audio_enhancements(self) -> Dict[str, bool]:
        """Validate that all audio enhancement features are working properly.
        
        Returns:
            Dict[str, bool]: Status of each audio enhancement feature
        """
        validation_results = {
            'level_monitoring': False,
            'gain_control': False,
            'system_boost': False,
            'audio_processing': False,
            'normalization': False
        }
        
        try:
            # Test level monitoring
            test_data = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
            rms, peak = self.calculate_audio_level(test_data)
            validation_results['level_monitoring'] = rms >= 0 and peak >= 0
            
            # Test gain control
            processed_data = self.apply_gain_and_processing(test_data)
            validation_results['gain_control'] = len(processed_data) == len(test_data)
            
            # Test system boost availability
            validation_results['system_boost'] = self._get_system_source_name(
                self.config.get('microphone_device')
            ) is not None
            
            # Test audio processing
            validation_results['audio_processing'] = True  # Basic processing always available
            
            # Test normalization
            try:
                test_audio = AudioSegment.silent(duration=100)  # 100ms of silence
                normalized = self.normalize_audio_segment(test_audio)
                validation_results['normalization'] = normalized is not None
            except Exception:
                validation_results['normalization'] = False
            
        except Exception as e:
            logging.debug(f"Error during audio enhancement validation: {e}")
        
        return validation_results
    
    def get_audio_enhancement_status(self) -> str:
        """Get a human-readable status of audio enhancements.
        
        Returns:
            str: Status summary
        """
        validation = self.validate_audio_enhancements()
        
        status_lines = []
        status_lines.append("🎙️ Audio Enhancement Status:")
        
        if validation['level_monitoring']:
            status_lines.append("   ✓ Real-time audio level monitoring")
        else:
            status_lines.append("   ✗ Real-time audio level monitoring")
        
        if validation['gain_control']:
            agc_status = "enabled" if self.auto_gain_enabled else "disabled"
            status_lines.append(f"   ✓ Software gain control (AGC: {agc_status})")
        else:
            status_lines.append("   ✗ Software gain control")
        
        if validation['system_boost']:
            boost_status = "enabled" if self.config.get('system_mic_boost', False) else "available"
            status_lines.append(f"   ✓ System microphone boost ({boost_status})")
        else:
            status_lines.append("   ✗ System microphone boost (not available)")
        
        if validation['normalization']:
            norm_status = "enabled" if self.normalize_audio else "available"
            status_lines.append(f"   ✓ Audio normalization ({norm_status})")
        else:
            status_lines.append("   ✗ Audio normalization")
        
        comp_status = "enabled" if self.compressor_enabled else "disabled"
        status_lines.append(f"   ✓ Audio compression ({comp_status})")
        
        # Current settings
        status_lines.append("")
        status_lines.append("🔊 Current Settings:")
        status_lines.append(f"   Software Gain: {self.current_gain:.1f}x")
        status_lines.append(f"   Gain Boost: {self.gain_boost_db:+.0f}dB")
        status_lines.append(f"   Target RMS: {self.target_rms:.2f}")
        status_lines.append(f"   Noise Gate: {self.noise_gate_threshold:.0f}dB")
        
        return "\n".join(status_lines)
    
    def advanced_silence_removal(self, audio_file: str) -> Optional[str]:
        """Apply advanced silence removal with configurable thresholds and speech preservation.
        
        Args:
            audio_file: Path to input audio file
            
        Returns:
            Optional[str]: Path to processed audio file, or original file if processing fails
        """
        try:
            # Get advanced silence removal configuration
            silence_config = self.config.get('advanced_silence_removal', {})
            
            if not silence_config.get('enabled', False):
                logging.debug("Advanced silence removal disabled, using basic processing")
                return self.process_audio(audio_file)
            
            # Load audio
            audio = AudioSegment.from_wav(audio_file)
            original_duration = len(audio)
            
            logging.info(f"Starting advanced silence removal on {original_duration/1000:.2f}s audio")
            
            # Apply each step of silence removal
            audio = self._remove_leading_silence(audio, silence_config)
            audio = self._remove_trailing_silence(audio, silence_config)
            audio = self._process_internal_silences(audio, silence_config)
            
            # Ensure minimum duration and quality
            audio = self._validate_processed_audio(audio, silence_config)
            
            # Save processed audio
            processed_file = audio_file.replace('.wav', '_silence_removed.wav')
            audio.export(processed_file, format="wav")
            
            new_duration = len(audio)
            reduction_percent = ((original_duration - new_duration) / original_duration) * 100
            
            logging.info(f"Advanced silence removal completed: {original_duration/1000:.2f}s → {new_duration/1000:.2f}s "
                        f"({reduction_percent:.1f}% reduction)")
            
            return processed_file
            
        except Exception as e:
            logging.error(f"Error in advanced silence removal: {e}")
            return self.process_audio(audio_file)  # Fallback to basic processing
    
    def _remove_leading_silence(self, audio: AudioSegment, config: Dict[str, Any]) -> AudioSegment:
        """Remove silence from the beginning of audio while preserving speech onset.
        
        Args:
            audio: Input audio segment
            config: Silence removal configuration
            
        Returns:
            AudioSegment: Audio with leading silence removed
        """
        try:
            threshold_db = config.get('leading_silence_threshold_db', -40)
            min_duration = config.get('min_leading_silence_duration', 0.1) * 1000  # Convert to ms
            padding_ms = config.get('preserve_speech_padding_ms', 100)
            
            # Find first non-silent segment
            for i in range(0, len(audio), 50):  # Check every 50ms
                chunk = audio[i:i+50]
                if chunk.dBFS > threshold_db:
                    # Found speech, preserve some padding before it
                    cut_point = max(0, i - padding_ms)
                    
                    # Only cut if there's significant leading silence
                    if cut_point > min_duration:
                        logging.debug(f"Removed {cut_point}ms leading silence")
                        return audio[cut_point:]
                    else:
                        return audio
            
            # If no speech found, return original
            return audio
            
        except Exception as e:
            logging.debug(f"Error removing leading silence: {e}")
            return audio
    
    def _remove_trailing_silence(self, audio: AudioSegment, config: Dict[str, Any]) -> AudioSegment:
        """Remove silence from the end of audio while preserving speech completion.
        
        Args:
            audio: Input audio segment
            config: Silence removal configuration
            
        Returns:
            AudioSegment: Audio with trailing silence removed
        """
        try:
            threshold_db = config.get('trailing_silence_threshold_db', -40)
            min_duration = config.get('min_trailing_silence_duration', 0.1) * 1000  # Convert to ms
            padding_ms = config.get('preserve_speech_padding_ms', 100)
            
            # Find last non-silent segment
            for i in range(len(audio) - 50, 0, -50):  # Check every 50ms backwards
                chunk = audio[i:i+50]
                if chunk.dBFS > threshold_db:
                    # Found speech, preserve some padding after it
                    cut_point = min(len(audio), i + 50 + padding_ms)
                    
                    # Only cut if there's significant trailing silence
                    silence_duration = len(audio) - cut_point
                    if silence_duration > min_duration:
                        logging.debug(f"Removed {silence_duration}ms trailing silence")
                        return audio[:cut_point]
                    else:
                        return audio
            
            # If no speech found, return original
            return audio
            
        except Exception as e:
            logging.debug(f"Error removing trailing silence: {e}")
            return audio
    
    def _process_internal_silences(self, audio: AudioSegment, config: Dict[str, Any]) -> AudioSegment:
        """Intelligently process internal silences while preserving natural speech rhythm.
        
        Args:
            audio: Input audio segment
            config: Silence removal configuration
            
        Returns:
            AudioSegment: Audio with internal silences processed
        """
        try:
            threshold_db = config.get('internal_silence_threshold_db', -35)
            max_reduction = config.get('max_internal_silence_reduction', 0.5)
            preserve_rhythm = config.get('preserve_natural_rhythm', True)
            min_chunk_duration = config.get('minimum_chunk_duration_ms', 200)
            aggressiveness = config.get('aggressiveness', 'moderate')
            
            # Adjust parameters based on aggressiveness
            if aggressiveness == 'conservative':
                min_silence_len = 1500  # 1.5 seconds
                keep_silence = 300      # Keep 300ms around speech
            elif aggressiveness == 'aggressive':
                min_silence_len = 800   # 0.8 seconds
                keep_silence = 150      # Keep 150ms around speech
            else:  # moderate
                min_silence_len = 1200  # 1.2 seconds
                keep_silence = 200      # Keep 200ms around speech
            
            # Split on silence with intelligent parameters
            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=threshold_db,
                keep_silence=keep_silence
            )
            
            if not chunks:
                logging.debug("No chunks found in internal silence processing")
                return audio
            
            # Filter out chunks that are too short (likely noise)
            valid_chunks = []
            for chunk in chunks:
                if len(chunk) >= min_chunk_duration:
                    valid_chunks.append(chunk)
                else:
                    logging.debug(f"Filtered out {len(chunk)}ms chunk (too short)")
            
            if not valid_chunks:
                logging.debug("No valid chunks after filtering")
                return audio
            
            # Combine chunks with intelligent spacing
            processed_audio = AudioSegment.empty()
            
            for i, chunk in enumerate(valid_chunks):
                processed_audio += chunk
                
                # Add intelligent pause between chunks (except for last chunk)
                if i < len(valid_chunks) - 1 and preserve_rhythm:
                    # Calculate pause based on speech content and natural rhythm
                    pause_duration = self._calculate_natural_pause(chunk, valid_chunks[i+1])
                    
                    if pause_duration > 0:
                        silence_gap = AudioSegment.silent(duration=pause_duration)
                        processed_audio += silence_gap
            
            return processed_audio
            
        except Exception as e:
            logging.debug(f"Error processing internal silences: {e}")
            return audio
    
    def _calculate_natural_pause(self, current_chunk: AudioSegment, next_chunk: AudioSegment) -> int:
        """Calculate natural pause duration between speech chunks.
        
        Args:
            current_chunk: Current speech segment
            next_chunk: Next speech segment
            
        Returns:
            int: Pause duration in milliseconds
        """
        try:
            # Base pause duration
            base_pause = 150  # 150ms
            
            # Analyze chunk characteristics for intelligent pausing
            current_rms = current_chunk.rms
            next_rms = next_chunk.rms
            
            # If there's a significant volume change, might indicate sentence boundary
            if abs(current_rms - next_rms) > 500:
                return base_pause + 50  # Add 50ms for sentence boundaries
            
            # For similar volume levels, use shorter pause
            return base_pause
            
        except Exception:
            return 150  # Default pause
    
    def _validate_processed_audio(self, audio: AudioSegment, config: Dict[str, Any]) -> AudioSegment:
        """Validate processed audio meets quality requirements.
        
        Args:
            audio: Processed audio segment
            config: Silence removal configuration
            
        Returns:
            AudioSegment: Validated audio segment
        """
        try:
            min_duration = config.get('minimum_chunk_duration_ms', 200)
            
            # Ensure minimum duration
            if len(audio) < min_duration:
                logging.warning(f"Processed audio too short ({len(audio)}ms), padding to minimum")
                padding_needed = min_duration - len(audio)
                padding = AudioSegment.silent(duration=padding_needed)
                audio = padding + audio + padding
            
            # Check for audio quality issues
            if audio.rms == 0:
                logging.warning("Processed audio has no content, this may indicate over-aggressive silence removal")
            
            return audio
            
        except Exception as e:
            logging.debug(f"Error validating processed audio: {e}")
            return audio
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


class WhisperTranscriber:
    """OpenAI Whisper transcription functionality."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Load Whisper model."""
        if not WHISPER_AVAILABLE or whisper is None:
            logging.error("Whisper is not installed. Install with: ./install-whisper.sh")
            print("\nERROR: Whisper backend not available")
            print("Install with: ./install-whisper.sh")
            print("Or switch to Vosk: python bin/setup_backend.py --backend vosk")
            sys.exit(1)

        try:
            model_name = self.config.get('whisper_model', 'base')
            logging.info(f"Loading Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            sys.exit(1)
    
    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file using Whisper."""
        if not self.model:
            logging.error("Whisper model not loaded")
            return None
        
        try:
            language = self.config.get('language')
            if language == 'auto':
                language = None
            
            logging.info(f"Transcribing audio: {audio_file}")
            
            # Enhanced parameters for French transcription
            transcribe_params = {
                'language': language,
                'fp16': False,
                'verbose': False,
                'beam_size': 5,
                'best_of': 5,
                'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                'compression_ratio_threshold': 2.4,
                'logprob_threshold': -1.0,
                'no_speech_threshold': 0.6
            }
            
            # French-specific optimizations
            if language == 'fr':
                transcribe_params.update({
                    'beam_size': 10,  # Better for French pronunciation variations
                    'best_of': 10,    # More candidates for French word selection
                    'temperature': (0.0, 0.1, 0.2, 0.3, 0.5),  # Lower temperature for French consistency
                    'compression_ratio_threshold': 2.2,  # Adjusted for French text density
                    'condition_on_previous_text': True,  # Better French context
                    'initial_prompt': "Transcription en français. Bonjour, comment allez-vous ?"  # French context primer
                })
            
            result = self.model.transcribe(audio_file, **transcribe_params)
            
            text = result['text'].strip()
            if text:
                logging.info(f"Transcription completed: {len(text)} characters")
                return text
            else:
                logging.warning("No text transcribed")
                return None
                
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return None


class DesktopIntegration:
    """Desktop integration for notifications and clipboard."""
    
    @staticmethod
    def send_notification(title: str, message: str, urgency: str = "normal") -> None:
        """Send desktop notification using notify-send."""
        try:
            cmd = ['notify-send', '-u', urgency, title, message]
            subprocess.run(cmd, check=True, capture_output=True)
            logging.debug(f"Notification sent: {title}")
        except Exception as e:
            logging.error(f"Error sending notification: {e}")
    
    @staticmethod
    def copy_to_clipboard(text: str) -> bool:
        """Copy text to clipboard."""
        try:
            pyperclip.copy(text)
            logging.info("Text copied to clipboard")
            return True
        except Exception as e:
            logging.error(f"Error copying to clipboard: {e}")
            return False
    
    @staticmethod
    def play_sound(sound_type: str, verbose: bool = True) -> None:
        """Play system sound for feedback using freedesktop sounds.
        
        Args:
            sound_type: Type of sound to play ('start' or 'stop')
            verbose: Whether to log sound feedback messages
        """
        try:
            sound_file = None
            if sound_type == "start":
                # Play window-attention sound for recording start
                sound_file = '/usr/share/sounds/freedesktop/stereo/window-attention.oga'
            elif sound_type == "stop":
                # Play power-unplug sound for recording stop
                sound_file = '/usr/share/sounds/freedesktop/stereo/power-unplug.oga'
            
            if sound_file and os.path.exists(sound_file):
                # Try multiple audio players for better compatibility
                audio_players = [
                    ['paplay', sound_file],
                    ['ffplay', '-nodisp', '-autoexit', sound_file],
                    ['aplay', sound_file]  # fallback for ALSA
                ]
                
                for player_cmd in audio_players:
                    try:
                        # Use subprocess.Popen for non-blocking playback
                        process = subprocess.Popen(
                            player_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        # Don't wait for completion to avoid blocking
                        if verbose:
                            logging.info(f"Sound feedback played: {sound_file}")
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                else:
                    logging.warning(f"No available audio player found for sound feedback")
            else:
                logging.warning(f"Sound file not found: {sound_file}")
                
        except Exception as e:
            logging.warning(f"Sound feedback not available: {e}")


class TranscriptionLogger:
    """Logging system for transcriptions."""
    
    def __init__(self, log_dir: str = "log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_transcription(self, text: str, audio_file: str, timestamp: str = None) -> None:
        """Log transcription with timestamp.
        
        Args:
            text: Transcribed text
            audio_file: Path to the audio file
            timestamp: Optional timestamp in 'YYYY-mm-dd HH:MM:SS' format.
                      If not provided, current time will be used.
        """
        try:
            # Use provided timestamp or generate new one if not provided
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"transcriptions_{date_str}.txt"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {os.path.basename(audio_file)}\n")
                f.write(f"Text: {text}\n")
                f.write("-" * 50 + "\n")
            
            logging.info(f"Transcription logged to {log_file}")
            
        except Exception as e:
            logging.error(f"Error logging transcription: {e}")


class HotkeyListeningTUI:
    """TUI that displays transcriptions from hotkey-triggered recordings."""

    def __init__(self, app):
        """Initialize with WhisperApp instance."""
        self.app = app
        self.running = True
        self.display_lock = threading.Lock()
        self.transcription_history = []

        # Hook into app's process_recording to capture transcriptions
        original_process = app.process_recording
        tui_self = self

        def process_with_display():
            """Wrapper that updates display after transcription."""
            # Call original process
            original_process()

            # Update display after processing
            time.sleep(0.5)  # Give time for transcription to complete
            tui_self.update_display()

        app.process_recording = process_with_display

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def update_display(self):
        """Update the TUI display."""
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

            # Get recent transcriptions from log
            has_transcriptions = False
            try:
                log_file = self.app.logger.get_today_log_file()
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Get last transcription
                        if lines:
                            has_transcriptions = True
                            last_line = lines[-1].strip()
                            if last_line:
                                # Parse: timestamp | text
                                parts = last_line.split('|', 1)
                                if len(parts) == 2:
                                    text = parts[1].strip()

                                    print("\n┌─ Latest Transcription " + "─" * 43 + "┐")
                                    # Word wrap
                                    words = text.split()
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

                        # Show last 10 transcriptions with full text
                        if len(lines) > 1:
                            print("\n─── Transcription History (scroll up to see more) ───")
                            history_count = min(10, len(lines) - 1)
                            for i, line in enumerate(lines[-history_count-1:-1], 1):
                                parts = line.strip().split('|', 1)
                                if len(parts) == 2:
                                    timestamp = parts[0].strip()
                                    text = parts[1].strip()
                                    # Show full text, wrapped for readability
                                    time_only = timestamp.split()[1] if ' ' in timestamp else timestamp
                                    print(f"\n  #{i} [{time_only}]")
                                    # Word wrap the full text
                                    words = text.split()
                                    line_text = "  "
                                    for word in words:
                                        if len(line_text + word) > 68:
                                            print(line_text)
                                            line_text = "  " + word + " "
                                        else:
                                            line_text += word + " "
                                    if line_text.strip():
                                        print(line_text)

            except Exception as e:
                logging.debug(f"Error reading log: {e}")

            if not self.app.audio_processor.is_recording and not has_transcriptions:
                print("\n[Waiting for recording... Press SUPER+A to start]")

            # Footer
            print("\n" + "─" * 70)
            print("  Press Ctrl+C to quit  |  Scroll up to copy previous transcriptions")
            print("=" * 70)

    def run(self):
        """Main display loop."""
        # Initial display
        self.update_display()

        # Monitor for changes and update display
        last_recording_state = False

        try:
            while self.running and self.app.is_running:
                time.sleep(0.3)

                # Update display when recording state changes
                current_state = self.app.audio_processor.is_recording
                if current_state != last_recording_state:
                    self.update_display()
                    last_recording_state = current_state

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.running = False
            self.app.is_running = False


class WhisperApp:
    """Main application class."""

    def __init__(self):
        self.config = WhisperConfig()
        self.audio_processor = AudioProcessor(self.config)

        # Use backend abstraction - check if backend is specified in config
        backend_type = self.config.get('backend', 'whisper')
        try:
            # Try to use new backend system if available
            from transcription_backends import BackendFactory
            self.transcriber = BackendFactory.create_backend(self.config.config)
            logging.info(f"Using {backend_type} backend via abstraction layer")

            # CRITICAL: Load the model
            if not self.transcriber.load_model():
                logging.error(f"Failed to load {backend_type} model")
                raise RuntimeError(f"Failed to load {backend_type} model")

        except ImportError as e:
            # Fallback to direct Whisper if backend module not available
            logging.warning(f"Backend abstraction not available: {e}, using direct Whisper")
            self.transcriber = WhisperTranscriber(self.config)

        self.desktop = DesktopIntegration()
        self.logger = TranscriptionLogger()
        self.is_running = False
        self.hotkey_listener = None
        self.daemon_mode = False
        self.last_transcription = ""  # For TUI display

        # Setup logging
        self.setup_logging()

        # Setup signal handlers for daemon mode
        signal.signal(signal.SIGUSR1, self.signal_toggle_recording)
        signal.signal(signal.SIGTERM, self.signal_shutdown)
        signal.signal(signal.SIGINT, self.signal_shutdown)
    
    def setup_logging(self) -> None:
        """Setup application logging."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path("log") / "whisper_app.log")
            ]
        )
    
    def signal_toggle_recording(self, signum, frame):
        """Signal handler for toggling recording (USR1)."""
        logging.info("Received toggle recording signal")
        self.toggle_recording()
    
    def signal_shutdown(self, signum, frame):
        """Signal handler for graceful shutdown (TERM, INT)."""
        logging.info("Received shutdown signal")
        self.is_running = False
    
    def configure_application(self) -> None:
        """Interactive configuration setup with compatibility validation."""
        current_backend = self.config.get('backend', 'vosk')
        print(f"\n=== Voice-to-Text Configuration ===")
        print(f"Current backend: {current_backend.upper()}")
        print(f"(To switch backend: python bin/setup_backend.py --backend vosk|whisper)\n")
        
        # Show audio system information
        audio_info = self.audio_processor.get_audio_system_info()
        print("🔍 Detected Audio System:")
        print(f"   Primary: {audio_info.primary_system.value}")
        print(f"   PulseAudio Compatible: {audio_info.pulseaudio_compatible}")
        print(f"   ALSA Compatible: {audio_info.alsa_compatible}")
        backend = self.audio_processor.audio_detector.get_recommended_audio_backend(audio_info)
        print(f"   Recommended Backend: {backend}")
        
        if audio_info.default_device:
            print(f"   Default Device: {audio_info.default_device.name}")
        print()
        
        # Test device compatibility first with timeout protection
        print("🔍 Testing device compatibility (this may take a few seconds)...")
        
        try:
            import signal
            import contextlib
            import sys
            import os
            
            # Set up a timeout for the compatibility testing
            def timeout_handler(signum, frame):
                raise TimeoutError("Device compatibility testing timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            # Comprehensive error suppression during testing - using file descriptor level
            original_stderr_fd = os.dup(2)  # Duplicate stderr file descriptor
            try:
                # Redirect stderr file descriptor to devnull to suppress C-level errors
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 2)  # Redirect stderr
                compatible_devices = self._test_device_compatibility()
            finally:
                # Always restore stderr file descriptor
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
                if 'devnull_fd' in locals():
                    os.close(devnull_fd)
            
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)
            
        except TimeoutError:
            print("⚠️  Device testing timed out. Using fallback device detection.")
            # Fallback: just get devices without compatibility testing
            devices = self.audio_processor.get_audio_devices(suppress_errors=True)
            compatible_devices = []
            for device in devices[:5]:  # Limit to first 5 devices
                compatible_devices.append({
                    'device': device,
                    'compatibility': {
                        'sample_rate': 16000,
                        'channels': 1,
                        'supported_rates': [16000]
                    }
                })
        except Exception as e:
            print(f"⚠️  Error during device testing: {e}")
            return
        
        if not compatible_devices:
            print("❌ No compatible audio devices found!")
            print("Please check your audio system and try again.")
            return
        
        # Audio device selection from compatible devices only
        print(f"\n✅ Compatible audio devices ({len(compatible_devices)} found):")
        for i, device_info in enumerate(compatible_devices):
            device = device_info['device']
            compatibility = device_info['compatibility']
            default_marker = " (DEFAULT)" if device.get('is_default', False) else ""
            system_info = f" [{device.get('system', 'unknown')}]"
            print(f"{i + 1}. {device['name'][:60]}{default_marker}{system_info}")
            
            # Show recommended configuration
            print(f"    Recommended: {compatibility['sample_rate']} Hz, {compatibility['channels']} channels")
            
            # Show supported sample rates if more than one
            supported_rates = compatibility.get('supported_rates', [])
            if len(supported_rates) > 1:
                rates_str = ', '.join(f"{rate}Hz" for rate in supported_rates)
                print(f"    Supports: {rates_str}")
            
            # Show if this matches user's current configuration
            current_rate = self.config.get('sample_rate', 44100)
            current_channels = self.config.get('channels', 2)
            if (compatibility['sample_rate'] == current_rate and 
                compatibility['channels'] == current_channels):
                print(f"    ✓ Matches your current config ({current_rate}Hz {current_channels}ch)")
            elif current_rate in supported_rates:
                print(f"    ✓ Supports your current rate ({current_rate}Hz)")
        
        while True:
            try:
                choice = input("\nSelect microphone device (number): ").strip()
                device_index = int(choice) - 1
                if 0 <= device_index < len(compatible_devices):
                    selected_device_info = compatible_devices[device_index]
                    selected_device = selected_device_info['device']
                    compatibility = selected_device_info['compatibility']
                    
                    # Store the device and its optimal configuration
                    device_id = selected_device.get('device_id') or selected_device.get('index')
                    self.config.set('microphone_device', device_id)
                    self.config.set('sample_rate', compatibility['sample_rate'])
                    self.config.set('channels', compatibility['channels'])
                    
                    print(f"✅ Selected: {selected_device['name']}")
                    print(f"   Device ID: {device_id}")
                    print(f"   Sample Rate: {compatibility['sample_rate']} Hz")
                    print(f"   Channels: {compatibility['channels']}")
                    
                    # Show additional compatibility info
                    supported_rates = compatibility.get('supported_rates', [])
                    if len(supported_rates) > 1:
                        rates_str = ', '.join(f"{rate}Hz" for rate in supported_rates)
                        print(f"   Also supports: {rates_str}")
                    
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Backend-specific model selection
        current_backend = self.config.get('backend', 'vosk')

        if current_backend == 'whisper':
            print("\nAvailable Whisper models:")
            models = ["tiny", "base", "small", "medium", "large"]
            print("  Recommended for your 4GB RAM system: tiny or base")
            for i, model in enumerate(models):
                print(f"{i + 1}. {model}")

            while True:
                try:
                    choice = input("Select Whisper model (number) [2 for base]: ").strip()
                    if not choice:
                        choice = "2"  # Default to base
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(models):
                        self.config.set('whisper_model', models[model_index])
                        print(f"Selected: {models[model_index]}")
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            # Vosk backend - no model selection needed (handled by setup_backend.py)
            vosk_model = self.config.get('vosk_model_path', 'models/vosk-model-small-en-us-0.15')
            print(f"\n✓ Using Vosk backend")
            print(f"  Model: {vosk_model}")
            print(f"  (To change Vosk model: python bin/setup_backend.py --interactive)")
        
        # Language selection
        print("\nLanguage options:")
        languages = [
            ("auto", "Auto-detect"),
            ("en", "English"),
            ("fr", "French"),
            ("es", "Spanish"),
            ("de", "German")
        ]
        for i, (code, name) in enumerate(languages):
            print(f"{i + 1}. {name}")
        
        while True:
            try:
                choice = input("Select language (number): ").strip()
                lang_index = int(choice) - 1
                if 0 <= lang_index < len(languages):
                    self.config.set('language', languages[lang_index][0])
                    print(f"Selected: {languages[lang_index][1]}")
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Audio sensitivity and gain configuration
        print("\n=== Audio Sensitivity Settings ===")
        
        # Current microphone volume
        current_volume = self.audio_processor.get_current_mic_volume()
        if current_volume is not None:
            print(f"Current system microphone volume: {current_volume:.0f}%")
        
        # Software gain configuration
        print("\nSoftware gain settings:")
        while True:
            try:
                current_gain = self.config.get('microphone_gain', 1.0)
                gain_input = input(f"Software gain multiplier (current: {current_gain:.1f}x, recommended: 1.0-4.0): ").strip()
                if not gain_input:  # Keep current value if empty
                    break
                gain = float(gain_input)
                if 0.1 <= gain <= 10.0:
                    self.config.set('microphone_gain', gain)
                    print(f"Software gain set to: {gain:.1f}x")
                    break
                else:
                    print("Gain must be between 0.1 and 10.0")
            except ValueError:
                print("Please enter a valid number")
        
        # Gain boost in dB
        while True:
            try:
                current_boost = self.config.get('gain_boost_db', 0)
                boost_input = input(f"Additional gain boost in dB (current: {current_boost:+.0f}dB, range: -20 to +20): ").strip()
                if not boost_input:  # Keep current value if empty
                    break
                boost_db = float(boost_input)
                if -20 <= boost_db <= 20:
                    self.config.set('gain_boost_db', boost_db)
                    print(f"Gain boost set to: {boost_db:+.0f}dB")
                    break
                else:
                    print("Gain boost must be between -20 and +20 dB")
            except ValueError:
                print("Please enter a valid number")
        
        # Auto Gain Control
        agc_input = input(f"Enable Auto Gain Control (current: {'yes' if self.config.get('auto_gain_control', True) else 'no'}) [y/n]: ").strip().lower()
        if agc_input in ['y', 'yes']:
            self.config.set('auto_gain_control', True)
            
            # Target RMS level for AGC
            while True:
                try:
                    current_target = self.config.get('target_rms_level', 0.1)
                    target_input = input(f"Target RMS level for AGC (current: {current_target:.2f}, range: 0.05-0.5): ").strip()
                    if not target_input:  # Keep current value if empty
                        break
                    target_rms = float(target_input)
                    if 0.05 <= target_rms <= 0.5:
                        self.config.set('target_rms_level', target_rms)
                        print(f"Target RMS level set to: {target_rms:.2f}")
                        break
                    else:
                        print("Target RMS must be between 0.05 and 0.5")
                except ValueError:
                    print("Please enter a valid number")
        elif agc_input in ['n', 'no']:
            self.config.set('auto_gain_control', False)
        
        # System microphone boost
        system_boost_input = input(f"Enable system-level microphone boost (current: {'yes' if self.config.get('system_mic_boost', False) else 'no'}) [y/n]: ").strip().lower()
        if system_boost_input in ['y', 'yes']:
            self.config.set('system_mic_boost', True)
            print("⚠️  System microphone boost enabled. This will increase your microphone volume at the OS level.")
        elif system_boost_input in ['n', 'no']:
            self.config.set('system_mic_boost', False)
        
        # Audio processing options
        print("\n=== Audio Processing Options ===")
        
        normalize_input = input(f"Enable audio normalization (current: {'yes' if self.config.get('normalize_audio', True) else 'no'}) [y/n]: ").strip().lower()
        if normalize_input in ['y', 'yes']:
            self.config.set('normalize_audio', True)
        elif normalize_input in ['n', 'no']:
            self.config.set('normalize_audio', False)
        
        compressor_input = input(f"Enable audio compression (current: {'yes' if self.config.get('compressor_enabled', True) else 'no'}) [y/n]: ").strip().lower()
        if compressor_input in ['y', 'yes']:
            self.config.set('compressor_enabled', True)
        elif compressor_input in ['n', 'no']:
            self.config.set('compressor_enabled', False)
        
        # Noise gate threshold
        while True:
            try:
                current_gate = self.config.get('noise_gate_threshold', -50)
                gate_input = input(f"Noise gate threshold in dB (current: {current_gate:.0f}dB, range: -80 to -20): ").strip()
                if not gate_input:  # Keep current value if empty
                    break
                gate_threshold = float(gate_input)
                if -80 <= gate_threshold <= -20:
                    self.config.set('noise_gate_threshold', gate_threshold)
                    print(f"Noise gate threshold set to: {gate_threshold:.0f}dB")
                    break
                else:
                    print("Noise gate threshold must be between -80 and -20 dB")
            except ValueError:
                print("Please enter a valid number")
        
        # Audio level display
        levels_input = input(f"Show real-time audio levels during recording (current: {'yes' if self.config.get('show_audio_levels', True) else 'no'}) [y/n]: ").strip().lower()
        if levels_input in ['y', 'yes']:
            self.config.set('show_audio_levels', True)
        elif levels_input in ['n', 'no']:
            self.config.set('show_audio_levels', False)
        
        # Manual mode level display setting
        if self.config.get('show_audio_levels', True):
            current_manual_setting = self.config.get('manual_mode_levels', 'minimal')
            print(f"\nManual mode audio level display options:")
            print("  full    - Show detailed levels (same as normal mode)")
            print("  minimal - Show simplified levels less frequently")
            print("  off     - No level display in manual mode")
            
            manual_input = input(f"Manual mode level display (current: {current_manual_setting}) [full/minimal/off]: ").strip().lower()
            if manual_input in ['full', 'minimal', 'off']:
                self.config.set('manual_mode_levels', manual_input)
            else:
                print(f"Keeping current setting: {current_manual_setting}")
        
        # Configure advanced silence removal
        print("\n⚡ Advanced Silence Removal Settings:")
        print("This feature automatically removes dead air and long pauses to improve efficiency.")
        
        silence_config = self.config.get('advanced_silence_removal', {})
        current_enabled = silence_config.get('enabled', True)
        current_aggressiveness = silence_config.get('aggressiveness', 'moderate')
        
        print(f"Current status: {'Enabled' if current_enabled else 'Disabled'}")
        print(f"Current aggressiveness: {current_aggressiveness}")
        
        # Ask about enabling/disabling
        enable_silence = input(f"Enable advanced silence removal? [y/n] (current: {'y' if current_enabled else 'n'}): ").strip().lower()
        if enable_silence in ['y', 'yes']:
            # Update existing config or use defaults
            silence_settings = self.config.get('advanced_silence_removal', {
                "enabled": True,
                "leading_silence_threshold_db": -40,
                "trailing_silence_threshold_db": -40,
                "internal_silence_threshold_db": -35,
                "min_leading_silence_duration": 0.1,
                "min_trailing_silence_duration": 0.1,
                "max_internal_silence_reduction": 0.5,
                "preserve_speech_padding_ms": 100,
                "aggressiveness": "moderate",
                "preserve_natural_rhythm": True,
                "minimum_chunk_duration_ms": 200
            })
            silence_settings['enabled'] = True
            
            # Ask about aggressiveness level
            print("\nAggressiveness levels:")
            print("  conservative - Removes only very long silences (1.5s+), preserves more natural pauses")
            print("  moderate     - Balanced silence removal (1.2s+), good for most use cases")
            print("  aggressive   - Removes shorter silences (0.8s+), maximum efficiency")
            
            aggressiveness = input(f"Choose aggressiveness level [conservative/moderate/aggressive] (current: {current_aggressiveness}): ").strip().lower()
            if aggressiveness in ['conservative', 'moderate', 'aggressive']:
                silence_settings['aggressiveness'] = aggressiveness
            
            self.config.set('advanced_silence_removal', silence_settings)
            print(f"✅ Advanced silence removal enabled with {silence_settings['aggressiveness']} settings")
            
        elif enable_silence in ['n', 'no']:
            silence_settings = self.config.get('advanced_silence_removal', {})
            silence_settings['enabled'] = False
            self.config.set('advanced_silence_removal', silence_settings)
            print("❌ Advanced silence removal disabled")
        else:
            print(f"Keeping current setting: {'Enabled' if current_enabled else 'Disabled'}")
        
        # Save configuration
        self.config.save_config()
        print("\nConfiguration saved successfully!")
        print("\n📝 Summary of audio settings:")
        print(f"   Software Gain: {self.config.get('microphone_gain', 1.0):.1f}x")
        print(f"   Gain Boost: {self.config.get('gain_boost_db', 0):+.0f}dB")
        print(f"   Auto Gain Control: {'Enabled' if self.config.get('auto_gain_control', True) else 'Disabled'}")
        print(f"   System Mic Boost: {'Enabled' if self.config.get('system_mic_boost', False) else 'Disabled'}")
        print(f"   Audio Normalization: {'Enabled' if self.config.get('normalize_audio', True) else 'Disabled'}")
        print(f"   Audio Compression: {'Enabled' if self.config.get('compressor_enabled', True) else 'Disabled'}")
        print(f"   Real-time Levels: {'Enabled' if self.config.get('show_audio_levels', True) else 'Disabled'}")
        if self.config.get('show_audio_levels', True):
            print(f"   Manual Mode Levels: {self.config.get('manual_mode_levels', 'minimal')}")
        
        # Show silence removal settings
        silence_config = self.config.get('advanced_silence_removal', {})
        if silence_config.get('enabled', False):
            aggressiveness = silence_config.get('aggressiveness', 'moderate')
            print(f"   Silence Removal: Enabled ({aggressiveness})")
        else:
            print(f"   Silence Removal: Disabled")
        print("\nYou can now run the application with: python bin/voice_transcriber.py")
        print("For testing audio levels, try: python bin/voice_transcriber.py --manual")
    
    def _test_device_compatibility(self) -> List[Dict[str, Any]]:
        """Test audio devices for compatibility with comprehensive sample rate and channel testing."""
        compatible_devices = []
        devices = self.audio_processor.get_audio_devices(suppress_errors=True)
        
        # Filter out problematic virtual devices that commonly cause errors
        filtered_devices = []
        skip_patterns = ['speex', 'upmix', 'spdif', 'default', 'jack']
        
        for device in devices:
            device_name = device.get('name', '').lower()
            # Skip virtual/problematic devices that typically don't work for recording
            if not any(pattern in device_name for pattern in skip_patterns):
                filtered_devices.append(device)
            # But keep hardware devices and the default device
            elif 'hw:' in device_name or device.get('is_default', False):
                filtered_devices.append(device)
        
        # Limit to first 8 devices to avoid excessive testing
        test_devices = filtered_devices[:8]
        print(f"Testing {len(test_devices)} audio devices for compatibility...")
        
        # Test common configurations that real hardware supports
        # Prioritize configurations based on common USB audio device capabilities
        test_configs = [
            (16000, 1),   # 16kHz mono (Whisper native)
            (44100, 1),   # CD quality mono (common for USB audio)
            (44100, 2),   # CD quality stereo (common for USB audio)
            (48000, 1),   # Professional mono
            (48000, 2),   # Professional stereo
            (16000, 2),   # 16kHz stereo
        ]
        
        for i, device in enumerate(test_devices):
            device_id = device.get('device_id') or device.get('index')
            device_index = self.audio_processor._get_pyaudio_device_index(device_id)
            
            if device_index is None:
                continue
            
            # Show progress indicator
            device_name = device['name'][:35] + '...' if len(device['name']) > 35 else device['name']
            print(f"Testing device {i+1}/{len(test_devices)}: {device_name}")
            
            # Test multiple configurations to find what works
            working_configs = []
            supported_rates = []
            best_config = None
            
            for sample_rate, channels in test_configs:
                if self.audio_processor._test_audio_config(device_index, sample_rate, channels):
                    working_configs.append((sample_rate, channels))
                    if sample_rate not in supported_rates:
                        supported_rates.append(sample_rate)
                    
                    # Choose best configuration (prefer existing user config, then common rates)
                    current_rate = self.config.get('sample_rate', 44100)
                    current_channels = self.config.get('channels', 2)
                    
                    if best_config is None:
                        best_config = (sample_rate, channels)
                    elif (sample_rate == current_rate and channels == current_channels):
                        # Exact match to user's current config - this is ideal
                        best_config = (sample_rate, channels)
                    elif sample_rate == current_rate and best_config[0] != current_rate:
                        # Same sample rate as user's config
                        best_config = (sample_rate, channels)
                    elif sample_rate == 44100 and best_config[0] not in [current_rate, 44100]:
                        # Prefer 44100Hz (common USB audio rate) over 16000Hz
                        best_config = (sample_rate, channels)
            
            if working_configs:
                best_rate, best_channels = best_config
                compatibility_info = {
                    'sample_rate': best_rate,
                    'channels': best_channels,
                    'supported_rates': sorted(supported_rates),
                    'all_configs': working_configs
                }
                
                compatible_devices.append({
                    'device': device,
                    'compatibility': compatibility_info
                })
                
                # Show detailed compatibility info
                config_summary = f"{best_rate}Hz {best_channels}ch"
                if len(supported_rates) > 1:
                    config_summary += f" (supports {len(supported_rates)} rates)"
                print(f"  ✓ Compatible - {config_summary}")
            else:
                print(f"  ✗ Not compatible")
        
        print(f"\n✓ Found {len(compatible_devices)} compatible devices")
        return compatible_devices
    
    def toggle_recording(self) -> None:
        """Toggle recording state."""
        if not self.audio_processor.is_recording:
            # Start recording
            if self.audio_processor.start_recording():
                if self.config.get('sound_feedback'):
                    verbose = self.config.get('sound_feedback_verbose', True)
                    self.desktop.play_sound("start", verbose)
                if self.config.get('notification_enabled'):
                    self.desktop.send_notification(
                        "", 
                        "🎙️", 
                        "normal"
                    )
        else:
            # Stop recording and process
            threading.Thread(target=self.process_recording, daemon=True).start()
    
    def process_recording(self) -> None:
        """Process the recorded audio and transcribe."""
        try:
            # Stop recording
            logging.info("Stopping recording...")
            recording_result = self.audio_processor.stop_recording()
            if not recording_result:
                logging.warning("No recording result returned")
                return

            # Unpack the tuple containing file path and timestamp
            audio_file, audio_timestamp = recording_result
            logging.info(f"Recording saved: {audio_file}")

            if self.config.get('sound_feedback'):
                verbose = self.config.get('sound_feedback_verbose', True)
                self.desktop.play_sound("stop", verbose)

            # Removed processing notification for minimal UI

            # Process audio with advanced silence removal
            logging.info("Processing audio (silence removal)...")
            processed_file = self.audio_processor.advanced_silence_removal(audio_file)
            if not processed_file:
                logging.warning("No audio detected after silence removal")
                # No notification for failed audio detection (minimal UI)
                return

            logging.info(f"Processed file: {processed_file}")

            # Transcribe
            logging.info("Starting transcription...")
            text = self.transcriber.transcribe_audio(processed_file)
            if text:
                logging.info(f"Transcription successful: {text}")
                self.last_transcription = text  # Store for TUI

                # Copy to clipboard
                self.desktop.copy_to_clipboard(text)

                # Log transcription with original audio file timestamp
                self.logger.log_transcription(text, audio_file, audio_timestamp)

                # Show notification
                if self.config.get('notification_enabled'):
                    # Show first few words with clipboard emoji
                    words = text.split()
                    preview = ' '.join(words[:6]) if len(words) > 6 else text
                    if len(words) > 6:
                        preview += "..."
                    self.desktop.send_notification(
                        "",
                        f"📋 > {preview}",
                        "normal"
                    )

                print(f"Transcribed: {text}")
            else:
                logging.warning("Transcription returned empty result")
                # No notification for failed speech detection (minimal UI)
                pass
            
            # Cleanup processed files
            try:
                if processed_file != audio_file:
                    os.remove(processed_file)
                # Optionally remove original recording
                # os.remove(audio_file)
            except Exception as e:
                logging.debug(f"Cleanup error: {e}")
                
        except Exception as e:
            logging.error(f"Error processing recording: {e}")
            self.desktop.send_notification(
                "Whisper Voice-to-Text", 
                f"Processing error: {str(e)}", 
                "critical"
            )
    
    def setup_hotkey(self) -> None:
        """Setup global hotkey listener with Wayland/Hyprland support."""
        try:
            # Import the Wayland-compatible hotkey handler
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from wayland_hotkey_handler import create_wayland_hotkey_handler
            
            def on_hotkey():
                self.toggle_recording()
            
            # Detect environment and use appropriate hotkey method
            if self._is_wayland_session():
                logging.info("Wayland session detected, using Wayland-compatible hotkey handler")
                self.hotkey_handler = create_wayland_hotkey_handler()
                success = self.hotkey_handler.setup_hotkey(on_hotkey)
                
                if success:
                    logging.info("Wayland-compatible hotkey (SUPER+A) registered successfully")
                else:
                    raise Exception("Failed to setup Wayland hotkey handler")
            else:
                # Fallback to pynput for X11 sessions
                logging.info("X11 session detected, using pynput hotkey handler")
                self.hotkey_listener = keyboard.GlobalHotKeys({
                    '<super>+a': on_hotkey
                })
                self.hotkey_listener.start()
                logging.info("Global hotkey (SUPER+A) registered with pynput")
            
        except Exception as e:
            logging.error(f"Error setting up hotkey: {e}")
            print("Warning: Global hotkey setup failed.")
            print("Available alternatives:")
            print("1. Run with --manual mode: python bin/voice_transcriber.py --manual")
            print("2. Use Hyprland keybind: bind = SUPER, a, exec, pkill -SIGUSR1 -f voice_transcriber.py")
            print("3. Check if your Wayland compositor supports global hotkeys")
    
    def _is_wayland_session(self) -> bool:
        """Detect if running under Wayland."""
        return (os.environ.get('WAYLAND_DISPLAY') is not None or 
                os.environ.get('XDG_SESSION_TYPE') == 'wayland' or
                os.environ.get('HYPRLAND_INSTANCE_SIGNATURE') is not None)
    
    def _resolve_hardware_device_name(self, device_name: str, device_index: int) -> str:
        """
        Resolve device aliases like 'sysdefault' to actual hardware device names.
        
        Args:
            device_name: The device name from PyAudio (might be an alias)
            device_index: PyAudio device index
            
        Returns:
            str: A more descriptive device name showing actual hardware
        """
        # If it's already a hardware device name, return as-is
        if device_name not in ['sysdefault', 'default', 'pulse', 'alsa']:
            return device_name
        
        try:
            # Get audio system info to determine approach
            audio_info = self.audio_processor.get_audio_system_info()
            
            # Method 1: Use PulseAudio/PipeWire commands to resolve default device
            if audio_info.primary_system in [AudioSystem.PIPEWIRE, AudioSystem.PULSEAUDIO]:
                hardware_name = self._resolve_pulse_default_device()
                if hardware_name:
                    return f"{hardware_name} (via {device_name})"
            
            # Method 2: Check if any detected audio devices match this PyAudio index
            devices = self.audio_processor.get_audio_devices()
            for device in devices:
                if device.get('index') == device_index and device.get('name', '').strip():
                    actual_name = device['name']
                    if actual_name != device_name:
                        return f"{actual_name} (via {device_name})"
            
            # Method 3: Try to get more info from PyAudio device properties
            try:
                device_info = self.audio_processor.audio.get_device_info_by_index(device_index)
                host_api_info = self.audio_processor.audio.get_host_api_info_by_index(device_info['hostApi'])
                host_api_name = host_api_info.get('name', 'Unknown')
                
                # For ALSA, try to get the card name
                if 'ALSA' in host_api_name:
                    card_name = self._get_alsa_card_name(device_index)
                    if card_name:
                        return f"{card_name} (via {device_name})"
                
            except Exception as e:
                logging.debug(f"Could not get extended device info: {e}")
            
            # Method 4: Use pactl to get default source info (for PulseAudio/PipeWire)
            default_source_info = self._get_default_source_description()
            if default_source_info:
                return f"{default_source_info} (via {device_name})"
            
        except Exception as e:
            logging.debug(f"Error resolving hardware device name: {e}")
        
        # If all methods fail, return original name
        return device_name
    
    def _resolve_pulse_default_device(self) -> Optional[str]:
        """Get the actual hardware device name behind the PulseAudio/PipeWire default source."""
        try:
            # Get default source name
            result = subprocess.run(['pactl', 'get-default-source'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode != 0:
                return None
            
            default_source = result.stdout.strip()
            
            # Get detailed info about the default source
            result = subprocess.run(['pactl', 'list', 'sources'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None
            
            # Parse the output to find our device
            current_source = None
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('Source #'):
                    current_source = None
                elif line.startswith('Name:') and default_source in line:
                    current_source = default_source
                elif current_source and line.startswith('Description:'):
                    # Extract description which usually contains the hardware name
                    description = line.replace('Description:', '').strip()
                    # Clean up common prefixes/suffixes
                    description = description.replace('Monitor of ', '')
                    return description
                elif current_source and line.startswith('device.description'):
                    # Alternative: get from properties
                    if '=' in line:
                        description = line.split('=', 1)[1].strip(' "')
                        return description
                        
        except Exception as e:
            logging.debug(f"Error resolving PulseAudio default device: {e}")
        
        return None
    
    def _get_alsa_card_name(self, device_index: int) -> Optional[str]:
        """Get ALSA card name for a device index."""
        try:
            # Try to map PyAudio device to ALSA card
            device_info = self.audio_processor.audio.get_device_info_by_index(device_index)
            device_name = device_info.get('name', '')
            
            # If device name contains card info, extract it
            if 'card' in device_name.lower():
                return device_name
            
            # Read ALSA cards info
            with open('/proc/asound/cards', 'r') as f:
                content = f.read()
            
            # Parse cards and return the first one (usually the default)
            for line in content.split('\n'):
                if line.strip() and not line.startswith(' '):
                    # Extract card name from line like: " 0 [PCH            ]: HDA-Intel - HDA Intel PCH"
                    if ']:' in line:
                        card_name = line.split(']:')[1].strip()
                        if ' - ' in card_name:
                            return card_name.split(' - ')[1]
                        return card_name
                        
        except Exception as e:
            logging.debug(f"Error getting ALSA card name: {e}")
        
        return None
    
    def _get_default_source_description(self) -> Optional[str]:
        """Get a more descriptive name for the default audio source."""
        try:
            # Try pactl info first
            result = subprocess.run(['pactl', 'info'], 
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Default Source:' in line:
                        source_name = line.split(':', 1)[1].strip()
                        
                        # Get description for this source
                        list_result = subprocess.run(['pactl', 'list', 'sources'], 
                                                   capture_output=True, text=True, timeout=5)
                        if list_result.returncode == 0:
                            lines = list_result.stdout.split('\n')
                            found_source = False
                            for i, list_line in enumerate(lines):
                                if f'Name: {source_name}' in list_line:
                                    found_source = True
                                elif found_source and 'Description:' in list_line:
                                    desc = list_line.split('Description:', 1)[1].strip()
                                    # Clean up the description
                                    if desc and desc != source_name:
                                        return desc
                                elif found_source and list_line.startswith('Source #'):
                                    # Moved to next source, stop looking
                                    break
                        
                        return source_name
                        
        except Exception as e:
            logging.debug(f"Error getting default source description: {e}")
        
        return None

    def _display_microphone_info(self) -> None:
        """Display detailed microphone information for debugging."""
        try:
            # Get validated configuration
            validated_config = self.audio_processor.get_validated_config()
            
            if not validated_config:
                print("⚠️  No valid microphone configuration available")
                return
            
            # Get current device information
            configured_device_id = self.config.get('microphone_device')
            device_index = validated_config['device_index']
            sample_rate = validated_config['sample_rate']
            channels = validated_config['channels']
            
            # Get device name from PyAudio
            device_name = "Unknown Device"
            device_info = None
            try:
                device_info = self.audio_processor.audio.get_device_info_by_index(device_index)
                device_name = device_info['name']
            except Exception as e:
                logging.debug(f"Could not get device info for index {device_index}: {e}")
            
            # Resolve hardware device name (this is the main enhancement)
            resolved_device_name = self._resolve_hardware_device_name(device_name, device_index)
            
            # Display main microphone info with resolved name
            print(f"Microphone: {resolved_device_name} (PyAudio index: {device_index})")
            print(f"Sample Rate: {sample_rate} Hz, Channels: {channels}")
            
            # Display device capabilities if available
            if device_info:
                max_input_channels = device_info.get('maxInputChannels', 'Unknown')
                default_sample_rate = int(device_info.get('defaultSampleRate', 0))
                print(f"Device Capabilities: Max {max_input_channels} input channels, Default {default_sample_rate} Hz")
            
            # Test and display supported sample rates
            supported_rates = []
            for rate in self.audio_processor.PREFERRED_SAMPLE_RATES:
                if self.audio_processor._test_audio_config(device_index, rate, channels):
                    supported_rates.append(rate)
            
            if supported_rates:
                rates_str = ', '.join(f"{rate} Hz" for rate in supported_rates[:5])  # Show first 5
                if len(supported_rates) > 5:
                    rates_str += f" (+{len(supported_rates) - 5} more)"
                print(f"Supported Sample Rates: {rates_str}")
            
            # Show audio system information
            audio_info = self.audio_processor.get_audio_system_info()
            backend = self.audio_processor.audio_detector.get_recommended_audio_backend(audio_info)
            print(f"Audio System: {audio_info.primary_system.value} (backend: {backend})")
            
            # Show configured device ID if different from PyAudio index
            if configured_device_id and str(configured_device_id) != str(device_index):
                print(f"Configured Device ID: {configured_device_id}")
            
        except Exception as e:
            print(f"⚠️  Error displaying microphone info: {e}")
            logging.error(f"Error in _display_microphone_info: {e}")
        
        print()  # Add empty line for spacing
    
    def run_audio_test_mode(self) -> None:
        """Run audio level testing mode without transcription."""
        print("\n=== Audio Level Testing Mode ===")
        
        # Display microphone information
        self._display_microphone_info()
        
        print("This mode will show real-time audio levels without transcription.")
        print("Use this to test your microphone sensitivity and adjust settings.")
        print("Press Enter to start/stop audio monitoring, 'quit' to exit.\n")
        
        # Show current audio settings
        print("🔊 Current Audio Settings:")
        print(f"   Software Gain: {self.config.get('microphone_gain', 1.0):.1f}x")
        print(f"   Gain Boost: {self.config.get('gain_boost_db', 0):+.0f}dB")
        print(f"   Auto Gain Control: {'ON' if self.config.get('auto_gain_control', True) else 'OFF'}")
        print(f"   Target RMS Level: {self.config.get('target_rms_level', 0.1):.2f}")
        print(f"   System Mic Boost: {'ON' if self.config.get('system_mic_boost', False) else 'OFF'}")
        print()
        
        # Show level legend
        print("📊 Audio Level Legend:")
        print("   [████████████████████] - Audio level bar")
        print("   Green zone: Good levels (-20dB to -10dB)")
        print("   Yellow zone: Loud levels (-10dB to -5dB)")
        print("   Red zone: Very loud levels (-5dB+)")
        print("   | - Peak level indicator")
        print("   Target: Speak normally and aim for green/yellow zones")
        print()
        
        is_monitoring = False
        
        while True:
            try:
                user_input = input("Press Enter to toggle audio monitoring (or 'quit' to exit): ").strip()
                if user_input.lower() == 'quit':
                    break
                
                if not is_monitoring:
                    # Start monitoring
                    if self.audio_processor.start_recording():
                        is_monitoring = True
                        print("✓ Audio monitoring started. Speak into your microphone...")
                        print("Press Enter again to stop monitoring.\n")
                    else:
                        print("✗ Failed to start audio monitoring")
                else:
                    # Stop monitoring
                    self.audio_processor.stop_recording()
                    is_monitoring = False
                    print("\n✓ Audio monitoring stopped.")
                    
                    # Show summary statistics
                    with self.audio_processor.level_lock:
                        if self.audio_processor.level_history:
                            avg_level = np.mean(self.audio_processor.level_history)
                            max_level = np.max(self.audio_processor.level_history)
                            avg_db = self.audio_processor.calculate_db_level(avg_level)
                            max_db = self.audio_processor.calculate_db_level(max_level)
                            
                            print(f"Session Summary:")
                            print(f"   Average Level: {avg_db:+5.1f}dB")
                            print(f"   Peak Level: {max_db:+5.1f}dB")
                            print(f"   Final Gain: {self.audio_processor.current_gain:.2f}x")
                            
                            # Recommendations
                            if avg_db < -30:
                                print("📈 Recommendation: Your voice is too quiet. Consider:")
                                print("     - Speaking louder or closer to the microphone")
                                print("     - Increasing software gain in configuration")
                                print("     - Enabling system microphone boost")
                            elif avg_db > -10:
                                print("📉 Recommendation: Your voice is very loud. Consider:")
                                print("     - Speaking more softly or moving away from microphone")
                                print("     - Reducing software gain in configuration")
                            else:
                                print("✓ Good audio levels detected!")
                    print()
                    
            except KeyboardInterrupt:
                if is_monitoring:
                    self.audio_processor.stop_recording()
                break
    
    def run_manual_mode(self) -> None:
        """Run in manual mode for testing or when hotkeys don't work."""
        self.manual_mode = True
        print("\n=== Manual Mode ===")
        
        # Display microphone information
        self._display_microphone_info()
        
        # Display audio enhancement status
        print(self.audio_processor.get_audio_enhancement_status())
        
        # Display level monitoring settings for manual mode
        manual_mode_setting = self.config.get('manual_mode_levels', 'minimal')
        print(f"Audio level display: {manual_mode_setting}")
        if manual_mode_setting != 'off':
            print("(Configure with: python bin/voice_transcriber.py --config)")
        print()
        
        print("Press Enter to start recording, Enter again to stop and transcribe.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("Press Enter to toggle recording (or 'quit' to exit): ").strip()
                if user_input.lower() == 'quit':
                    break
                self.toggle_recording()
            except KeyboardInterrupt:
                break
    
    def run_daemon_mode(self) -> None:
        """Run in daemon mode for background operation."""
        self.daemon_mode = True
        self.is_running = True
        
        # Write PID file
        pidfile = Path(".whisper_daemon.pid")
        with open(pidfile, 'w') as f:
            f.write(str(os.getpid()))
        
        logging.info("Whisper daemon started")
        self.desktop.send_notification(
            "Whisper Voice-to-Text",
            "Daemon started. Send USR1 signal to toggle recording.",
            "normal"
        )
        
        try:
            # Keep the daemon running
            while self.is_running:
                time.sleep(1)
        finally:
            # Clean up PID file
            if pidfile.exists():
                pidfile.unlink()
    
    def run(self, manual_mode: bool = False, daemon_mode: bool = False) -> None:
        """Run the main application."""
        try:
            self.is_running = True
            
            if daemon_mode:
                self.run_daemon_mode()
            elif manual_mode:
                self.run_manual_mode()
            else:
                self.setup_hotkey()
                
                print("Whisper Voice-to-Text is running...")
                print("Press SUPER+A to start/stop recording")
                print("Press Ctrl+C to exit")
                
                # Keep the application running
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def run_tui_mode(self) -> None:
        """Run application with TUI display and hotkey listening."""
        print("Launching TUI with hotkey support...")
        print("Press SUPER+A to start/stop recording")
        print()

        # Mark app as running
        self.is_running = True

        # Setup hotkey handler (using signals for Wayland/Sway)
        self.setup_hotkey()

        # Start TUI display in main thread
        try:
            tui = HotkeyListeningTUI(self)
            tui.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up application resources."""
        self.is_running = False
        
        # Clean up hotkey handlers
        if hasattr(self, 'hotkey_listener') and self.hotkey_listener:
            self.hotkey_listener.stop()
        
        if hasattr(self, 'hotkey_handler') and self.hotkey_handler:
            self.hotkey_handler.cleanup()
        
        self.audio_processor.cleanup()
        logging.info("Application shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Voice-to-Text application using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/voice_transcriber.py              # Run with global hotkey
  python bin/voice_transcriber.py --config     # Configure the application
  python bin/voice_transcriber.py --manual     # Run in manual mode
  python bin/voice_transcriber.py --test-audio # Test audio levels and sensitivity
  python bin/voice_transcriber.py --daemon     # Run as background daemon
  python bin/audio_level_tuner.py              # Interactive audio tuning utility

Audio Enhancement Features:
  - Real-time audio level monitoring with visual feedback
  - Automatic gain control (AGC) for consistent levels
  - Software gain boost and system microphone boost
  - Audio normalization and compression
  - Noise gate and advanced audio processing
  - Interactive configuration for optimal sensitivity
        """
    )
    
    parser.add_argument(
        '--config', 
        action='store_true',
        help='Run configuration setup'
    )
    
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Run in manual mode (useful for testing or when hotkeys don\'t work)'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon (responds to USR1 signal for recording toggle)'
    )
    
    parser.add_argument(
        '--test-audio',
        action='store_true',
        help='Test audio levels and sensitivity (shows real-time levels without transcription)'
    )

    parser.add_argument(
        '--tui',
        action='store_true',
        help='Run with interactive TUI dashboard (3-panel interface with live monitoring)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Whisper Voice-to-Text 1.0.0'
    )
    
    args = parser.parse_args()

    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    app = WhisperApp()

    if args.config:
        app.configure_application()
    elif args.test_audio:
        app.run_audio_test_mode()
    elif args.tui:
        # Launch TUI mode
        app.run_tui_mode()
    else:
        app.run(manual_mode=args.manual, daemon_mode=args.daemon)


if __name__ == '__main__':
    main()