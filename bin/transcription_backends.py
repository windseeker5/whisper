#!/usr/bin/env python3
"""
Transcription Backend Abstraction Layer

Provides a unified interface for different speech-to-text backends (Whisper, Vosk, etc.)
allowing easy switching and comparison between transcription engines.

Author: Python DevOps Automation Specialist
Compatible: Arch Linux, Wayland/Sway
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        """Load the transcription model.

        Returns:
            bool: True if model loaded successfully
        """
        pass

    @abstractmethod
    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio file to text.

        Args:
            audio_file: Path to audio file (WAV format)

        Returns:
            str: Transcribed text or None if failed
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model to free memory."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            dict: Model information (name, size, language, etc.)
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of this backend.

        Returns:
            str: Backend name
        """
        pass


class WhisperBackend(TranscriptionBackend):
    """OpenAI Whisper transcription backend."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Whisper backend.

        Args:
            config: Configuration with 'whisper_model' and 'language' keys
        """
        super().__init__(config)
        self.model_name = config.get('whisper_model', 'base')
        self.language = config.get('language', 'auto')

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "Whisper"

    def load_model(self) -> bool:
        """Load Whisper model."""
        try:
            import whisper
            logging.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            self.is_loaded = True
            logging.info("Whisper model loaded successfully")
            return True
        except ImportError:
            logging.error("Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            return False

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio using Whisper."""
        if not self.model or not self.is_loaded:
            logging.error("Whisper model not loaded")
            return None

        try:
            language = None if self.language == 'auto' else self.language

            logging.info(f"Transcribing with Whisper: {audio_file}")

            # Enhanced transcription parameters
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
                    'beam_size': 10,
                    'best_of': 10,
                    'temperature': (0.0, 0.1, 0.2, 0.3, 0.5),
                    'compression_ratio_threshold': 2.2,
                    'condition_on_previous_text': True,
                    'initial_prompt': "Transcription en français. Bonjour, comment allez-vous ?"
                })

            result = self.model.transcribe(audio_file, **transcribe_params)
            text = result['text'].strip()

            if text:
                logging.info(f"Whisper transcription completed: {len(text)} characters")
                return text
            else:
                logging.warning("No text transcribed by Whisper")
                return None

        except Exception as e:
            logging.error(f"Error transcribing with Whisper: {e}")
            return None

    def unload_model(self) -> None:
        """Unload Whisper model."""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logging.info("Whisper model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get Whisper model information."""
        model_sizes = {
            'tiny': {'params': '39M', 'size': '39 MB', 'speed': 'Very Fast'},
            'base': {'params': '74M', 'size': '74 MB', 'speed': 'Fast'},
            'small': {'params': '244M', 'size': '244 MB', 'speed': 'Medium'},
            'medium': {'params': '769M', 'size': '769 MB', 'speed': 'Slow'},
            'large': {'params': '1550M', 'size': '1550 MB', 'speed': 'Very Slow'}
        }

        info = model_sizes.get(self.model_name, {'params': 'Unknown', 'size': 'Unknown', 'speed': 'Unknown'})

        return {
            'backend': 'Whisper',
            'model': self.model_name,
            'parameters': info['params'],
            'size': info['size'],
            'speed': info['speed'],
            'language': self.language,
            'loaded': self.is_loaded
        }


class VoskBackend(TranscriptionBackend):
    """Vosk offline speech recognition backend (lightweight, fast)."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Vosk backend.

        Args:
            config: Configuration with 'vosk_model_path' and 'language' keys
        """
        super().__init__(config)
        self.model_path = config.get('vosk_model_path', './models/vosk-model-small-en-us-0.15')
        self.language = config.get('language', 'en')
        self.sample_rate = config.get('sample_rate', 16000)
        self.recognizer = None

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "Vosk"

    def load_model(self) -> bool:
        """Load Vosk model."""
        try:
            from vosk import Model, KaldiRecognizer
            import wave

            model_path = Path(self.model_path)
            if not model_path.exists():
                logging.error(f"Vosk model not found at: {model_path}")
                logging.info("Download models from: https://alphacephei.com/vosk/models")
                logging.info(f"Example: wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                logging.info(f"         unzip vosk-model-small-en-us-0.15.zip -d models/")
                return False

            logging.info(f"Loading Vosk model from: {model_path}")
            self.model = Model(str(model_path))
            self.is_loaded = True
            logging.info("Vosk model loaded successfully")
            return True

        except ImportError:
            logging.error("Vosk not installed. Install with: pip install vosk")
            return False
        except Exception as e:
            logging.error(f"Error loading Vosk model: {e}")
            return False

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio using Vosk."""
        if not self.model or not self.is_loaded:
            logging.error("Vosk model not loaded")
            return None

        try:
            from vosk import KaldiRecognizer
            import wave

            logging.info(f"Transcribing with Vosk: {audio_file}")

            # Open audio file
            wf = wave.open(audio_file, "rb")

            # Verify audio format
            if wf.getnchannels() != 1:
                logging.warning(f"Vosk expects mono audio, got {wf.getnchannels()} channels")

            # Create recognizer with correct sample rate
            sample_rate = wf.getframerate()
            recognizer = KaldiRecognizer(self.model, sample_rate)
            recognizer.SetWords(True)

            # Process audio in chunks
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if 'text' in result and result['text']:
                        results.append(result['text'])

            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            if 'text' in final_result and final_result['text']:
                results.append(final_result['text'])

            wf.close()

            # Combine all results
            text = ' '.join(results).strip()

            if text:
                logging.info(f"Vosk transcription completed: {len(text)} characters")
                return text
            else:
                logging.warning("No text transcribed by Vosk")
                return None

        except Exception as e:
            logging.error(f"Error transcribing with Vosk: {e}")
            return None

    def unload_model(self) -> None:
        """Unload Vosk model."""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            if self.recognizer:
                del self.recognizer
                self.recognizer = None
            logging.info("Vosk model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Get Vosk model information."""
        model_path = Path(self.model_path)
        model_name = model_path.name if model_path.exists() else "Not found"

        # Estimate size based on model name
        size_estimate = "~50 MB"
        if 'large' in model_name.lower():
            size_estimate = "~200-400 MB"
        elif 'small' in model_name.lower():
            size_estimate = "~40-50 MB"

        return {
            'backend': 'Vosk',
            'model': model_name,
            'path': str(model_path),
            'size': size_estimate,
            'speed': 'Very Fast',
            'language': self.language,
            'loaded': self.is_loaded,
            'sample_rate': self.sample_rate
        }


class BackendFactory:
    """Factory for creating transcription backends."""

    @staticmethod
    def create_backend(config: Dict[str, Any]) -> TranscriptionBackend:
        """Create a transcription backend based on configuration.

        Args:
            config: Configuration dictionary with 'backend' key

        Returns:
            TranscriptionBackend: Instantiated backend

        Raises:
            ValueError: If backend type is unknown
        """
        backend_type = config.get('backend', 'whisper').lower()

        if backend_type == 'whisper':
            return WhisperBackend(config)
        elif backend_type == 'vosk':
            return VoskBackend(config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}. Supported: whisper, vosk")

    @staticmethod
    def list_available_backends() -> List[str]:
        """List all available backend types.

        Returns:
            list: Available backend names
        """
        backends = []

        # Check if Whisper is available
        try:
            import whisper
            backends.append('whisper')
        except ImportError:
            pass

        # Check if Vosk is available
        try:
            import vosk
            backends.append('vosk')
        except ImportError:
            pass

        return backends

    @staticmethod
    def get_recommended_backend(ram_gb: float = 4.0) -> str:
        """Get recommended backend based on available RAM.

        Args:
            ram_gb: Available RAM in GB

        Returns:
            str: Recommended backend name
        """
        if ram_gb < 6:
            return 'vosk'  # Lightweight for systems with limited RAM
        else:
            return 'whisper'  # Better accuracy for systems with more RAM


if __name__ == "__main__":
    # Test the backends
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Available backends:", BackendFactory.list_available_backends())
    print("Recommended backend for 4GB RAM:", BackendFactory.get_recommended_backend(4.0))
    print("Recommended backend for 8GB RAM:", BackendFactory.get_recommended_backend(8.0))
