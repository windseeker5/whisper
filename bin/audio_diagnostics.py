#!/home/kdresdell/Documents/DEV/whisper/venv/bin/python
"""
Audio Diagnostics Tool
======================

This script diagnoses audio recording quality issues in the whisper voice transcription system.
It identifies problems like sample rate mismatches, timing issues, gain problems, and other
audio pipeline issues that can cause poor transcription quality.

Usage:
    python bin/audio_diagnostics.py [--analyze-file FILE] [--test-recording] [--fix-config]

Author: Claude Code - DevOps Automation Specialist
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pyaudio
import wave
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'log' / 'audio_diagnostics.log')
    ]
)

class AudioDiagnostics:
    """Comprehensive audio diagnostics for voice recording quality issues."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize audio diagnostics tool."""
        self.project_root = project_root
        self.config_path = config_path or str(project_root / 'config' / 'whisper_config.json')
        self.config = self._load_config()
        self.audio = None
        self.issues_found = []
        self.recommendations = []
        
    def _load_config(self) -> dict:
        """Load whisper configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self) -> None:
        """Save updated configuration."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info("Configuration saved successfully")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def analyze_recorded_file(self, file_path: str) -> Dict:
        """Analyze a recorded audio file for quality issues."""
        logging.info(f"Analyzing audio file: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return {}
        
        analysis = {
            'file_path': file_path,
            'issues': [],
            'stats': {},
            'recommendations': []
        }
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = len(audio_data) / sample_rate
            
            # Basic file stats
            analysis['stats'] = {
                'duration': duration,
                'sample_rate': sample_rate,
                'samples': len(audio_data),
                'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0],
                'file_size': os.path.getsize(file_path)
            }
            
            # Analyze audio quality issues
            self._check_sample_rate_issues(analysis, sample_rate)
            self._check_audio_levels(analysis, audio_data)
            self._check_noise_and_distortion(analysis, audio_data, sample_rate)
            self._check_timing_issues(analysis, audio_data, sample_rate)
            self._check_dynamic_range(analysis, audio_data)
            
        except Exception as e:
            logging.error(f"Error analyzing file: {e}")
            analysis['issues'].append(f"Analysis failed: {e}")
        
        return analysis
    
    def _check_sample_rate_issues(self, analysis: Dict, actual_rate: int) -> None:
        """Check for sample rate related issues."""
        expected_rate = self.config.get('sample_rate', 44100)
        
        if actual_rate != expected_rate:
            issue = f"Sample rate mismatch: file={actual_rate}Hz, config={expected_rate}Hz"
            analysis['issues'].append(issue)
            analysis['recommendations'].append(
                f"Update config sample_rate to {actual_rate} or ensure recording uses {expected_rate}Hz"
            )
        
        # Check if sample rate is appropriate for speech
        if actual_rate > 48000:
            analysis['issues'].append(f"Unnecessarily high sample rate ({actual_rate}Hz) for speech")
            analysis['recommendations'].append("Consider using 16kHz or 44.1kHz for speech recording")
        elif actual_rate < 16000:
            analysis['issues'].append(f"Sample rate too low ({actual_rate}Hz) for quality speech")
            analysis['recommendations'].append("Use at least 16kHz sample rate for speech")
    
    def _check_audio_levels(self, analysis: Dict, audio_data: np.ndarray) -> None:
        """Check audio levels for clipping, low volume, etc."""
        # Calculate RMS and peak levels
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_level = np.max(np.abs(audio_data))
        
        analysis['stats']['rms_level'] = float(rms_level)
        analysis['stats']['peak_level'] = float(peak_level)
        analysis['stats']['rms_db'] = float(20 * np.log10(rms_level + 1e-10))
        analysis['stats']['peak_db'] = float(20 * np.log10(peak_level + 1e-10))
        
        # Check for clipping (values at or near ±1.0)
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_data) >= clipping_threshold)
        if clipped_samples > 0:
            clipping_percentage = (clipped_samples / len(audio_data)) * 100
            analysis['issues'].append(f"Audio clipping detected: {clipping_percentage:.2f}% of samples")
            analysis['recommendations'].append("Reduce microphone gain or input volume")
        
        # Check for too low volume
        if rms_level < 0.01:  # -40dB
            analysis['issues'].append(f"Audio level too low: RMS={analysis['stats']['rms_db']:.1f}dB")
            analysis['recommendations'].append("Increase microphone gain or speak closer to microphone")
        
        # Check for too high volume (risk of distortion)
        if rms_level > 0.3:  # -10dB
            analysis['issues'].append(f"Audio level too high: RMS={analysis['stats']['rms_db']:.1f}dB")
            analysis['recommendations'].append("Reduce microphone gain to prevent distortion")
    
    def _check_noise_and_distortion(self, analysis: Dict, audio_data: np.ndarray, sample_rate: int) -> None:
        """Check for noise and distortion issues."""
        # Calculate noise floor (lowest 10th percentile of absolute values)
        sorted_abs = np.sort(np.abs(audio_data))
        noise_floor = sorted_abs[int(len(sorted_abs) * 0.1)]
        analysis['stats']['noise_floor_db'] = float(20 * np.log10(noise_floor + 1e-10))
        
        # Check for high noise floor
        if noise_floor > 0.005:  # -46dB
            analysis['issues'].append(f"High noise floor: {analysis['stats']['noise_floor_db']:.1f}dB")
            analysis['recommendations'].append("Check for electrical noise, enable noise reduction")
        
        # Check for DC offset
        dc_offset = np.mean(audio_data)
        if abs(dc_offset) > 0.01:
            analysis['issues'].append(f"DC offset detected: {dc_offset:.4f}")
            analysis['recommendations'].append("Enable DC offset removal in audio processing")
        
        # Check for harmonic distortion (simplified check)
        # Calculate spectral characteristics
        fft = np.fft.rfft(audio_data)
        magnitude_spectrum = np.abs(fft)
        
        # Check for unusual spectral peaks that might indicate distortion
        peak_indices = np.where(magnitude_spectrum > np.percentile(magnitude_spectrum, 99))[0]
        if len(peak_indices) > 10:  # Many peaks might indicate distortion
            analysis['issues'].append("Possible harmonic distortion detected in frequency spectrum")
            analysis['recommendations'].append("Reduce input gain to prevent distortion")
    
    def _check_timing_issues(self, analysis: Dict, audio_data: np.ndarray, sample_rate: int) -> None:
        """Check for timing-related issues that could cause 'accelerated' sound."""
        # This is a complex issue - look for irregularities in the audio stream
        
        # Check for buffer underruns/overruns by looking at sudden amplitude changes
        diff = np.diff(audio_data)
        large_jumps = np.where(np.abs(diff) > 0.1)[0]
        
        if len(large_jumps) > len(audio_data) * 0.001:  # More than 0.1% of samples
            analysis['issues'].append("Frequent amplitude discontinuities detected")
            analysis['recommendations'].append("Check for buffer timing issues, reduce chunk size")
        
        # Check for unusual frequency content that might indicate timing issues
        duration = len(audio_data) / sample_rate
        if duration < 1.0:  # Very short recordings might indicate timing problems
            analysis['issues'].append(f"Unusually short recording duration: {duration:.2f}s")
            analysis['recommendations'].append("Check recording trigger logic and minimum duration settings")
    
    def _check_dynamic_range(self, analysis: Dict, audio_data: np.ndarray) -> None:
        """Check dynamic range of the audio."""
        # Calculate dynamic range
        rms_level = np.sqrt(np.mean(audio_data**2))
        peak_level = np.max(np.abs(audio_data))
        
        if peak_level > 0:
            crest_factor = peak_level / rms_level
            analysis['stats']['crest_factor'] = float(crest_factor)
            
            # Check for over-compression (low crest factor)
            if crest_factor < 2.0:
                analysis['issues'].append(f"Audio appears over-compressed: crest factor={crest_factor:.1f}")
                analysis['recommendations'].append("Disable aggressive compression or normalization")
    
    def test_live_recording(self, duration: float = 5.0) -> Dict:
        """Test live recording to identify real-time audio issues."""
        logging.info(f"Testing live recording for {duration} seconds...")
        
        test_result = {
            'duration_requested': duration,
            'duration_actual': 0,
            'issues': [],
            'stats': {},
            'recommendations': []
        }
        
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Get device info
            device_id = self.config.get('microphone_device', 'pyaudio:0')
            device_index = int(device_id.split(':')[1]) if ':' in device_id else 0
            
            sample_rate = self.config.get('sample_rate', 44100)
            channels = self.config.get('channels', 1)
            chunk_size = self.config.get('chunk_size', 1024)
            
            # Test audio configuration
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk_size
                )
            except Exception as e:
                test_result['issues'].append(f"Failed to open audio stream: {e}")
                return test_result
            
            # Record audio
            frames = []
            start_time = time.time()
            
            try:
                for _ in range(int(sample_rate / chunk_size * duration)):
                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        frames.append(data)
                    except Exception as e:
                        test_result['issues'].append(f"Audio read error: {e}")
                        break
                
                actual_duration = time.time() - start_time
                test_result['duration_actual'] = actual_duration
                
                # Check timing accuracy
                timing_error = abs(actual_duration - duration) / duration
                if timing_error > 0.1:  # More than 10% timing error
                    test_result['issues'].append(
                        f"Timing inaccuracy: requested={duration:.2f}s, actual={actual_duration:.2f}s"
                    )
                    test_result['recommendations'].append("Check system audio timing, consider different chunk size")
                
            finally:
                stream.stop_stream()
                stream.close()
            
            # Analyze recorded data
            if frames:
                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                # Run quality analysis
                temp_analysis = {'stats': {}, 'issues': [], 'recommendations': []}
                self._check_audio_levels(temp_analysis, audio_data)
                self._check_noise_and_distortion(temp_analysis, audio_data, sample_rate)
                
                test_result['stats'].update(temp_analysis['stats'])
                test_result['issues'].extend(temp_analysis['issues'])
                test_result['recommendations'].extend(temp_analysis['recommendations'])
                
        except Exception as e:
            test_result['issues'].append(f"Recording test failed: {e}")
        finally:
            if self.audio:
                self.audio.terminate()
        
        return test_result
    
    def diagnose_configuration(self) -> Dict:
        """Diagnose current audio configuration for potential issues."""
        logging.info("Diagnosing audio configuration...")
        
        diagnosis = {
            'config_issues': [],
            'device_issues': [],
            'recommendations': []
        }
        
        # Check microphone gain settings
        mic_gain = self.config.get('microphone_gain', 1.0)
        if mic_gain > 2.0:
            diagnosis['config_issues'].append(f"High microphone gain: {mic_gain}")
            diagnosis['recommendations'].append("Reduce microphone_gain to prevent distortion")
        
        # Check auto gain control
        if self.config.get('auto_gain_control', False):
            diagnosis['config_issues'].append("Auto gain control enabled")
            diagnosis['recommendations'].append("Disable auto_gain_control for consistent levels")
        
        # Check chunk size
        chunk_size = self.config.get('chunk_size', 1024)
        sample_rate = self.config.get('sample_rate', 44100)
        buffer_duration = chunk_size / sample_rate * 1000  # milliseconds
        
        if buffer_duration < 10:
            diagnosis['config_issues'].append(f"Very small buffer: {buffer_duration:.1f}ms")
            diagnosis['recommendations'].append("Increase chunk_size to reduce CPU overhead")
        elif buffer_duration > 100:
            diagnosis['config_issues'].append(f"Large buffer: {buffer_duration:.1f}ms")
            diagnosis['recommendations'].append("Reduce chunk_size for better responsiveness")
        
        # Check device compatibility
        try:
            device_id = self.config.get('microphone_device', 'pyaudio:0')
            device_index = int(device_id.split(':')[1]) if ':' in device_id else 0
            
            self.audio = pyaudio.PyAudio()
            device_info = self.audio.get_device_info_by_index(device_index)
            
            # Check if device sample rate matches config
            device_rate = int(device_info['defaultSampleRate'])
            config_rate = self.config.get('sample_rate', 44100)
            
            if device_rate != config_rate:
                diagnosis['device_issues'].append(
                    f"Device rate ({device_rate}Hz) != config rate ({config_rate}Hz)"
                )
                diagnosis['recommendations'].append(f"Set sample_rate to {device_rate} for optimal compatibility")
            
            self.audio.terminate()
            
        except Exception as e:
            diagnosis['device_issues'].append(f"Device check failed: {e}")
        
        return diagnosis
    
    def fix_common_issues(self) -> bool:
        """Automatically fix common audio configuration issues."""
        logging.info("Attempting to fix common audio issues...")
        
        fixed_issues = []
        
        # Fix 1: Set optimal sample rate for the device
        try:
            device_id = self.config.get('microphone_device', 'pyaudio:0')
            device_index = int(device_id.split(':')[1]) if ':' in device_id else 0
            
            self.audio = pyaudio.PyAudio()
            device_info = self.audio.get_device_info_by_index(device_index)
            optimal_rate = int(device_info['defaultSampleRate'])
            
            if optimal_rate != self.config.get('sample_rate'):
                self.config['sample_rate'] = optimal_rate
                fixed_issues.append(f"Updated sample_rate to device optimal: {optimal_rate}Hz")
            
            self.audio.terminate()
        except Exception as e:
            logging.warning(f"Could not optimize sample rate: {e}")
        
        # Fix 2: Reduce microphone gain if too high
        if self.config.get('microphone_gain', 1.0) > 1.5:
            self.config['microphone_gain'] = 1.0
            fixed_issues.append("Reduced microphone_gain to 1.0")
        
        # Fix 3: Disable problematic features
        if self.config.get('auto_gain_control', False):
            self.config['auto_gain_control'] = False
            fixed_issues.append("Disabled auto_gain_control")
        
        if self.config.get('normalize_audio', False):
            self.config['normalize_audio'] = False
            fixed_issues.append("Disabled normalize_audio")
        
        # Fix 4: Set reasonable chunk size
        sample_rate = self.config.get('sample_rate', 44100)
        optimal_chunk = int(sample_rate * 0.025)  # 25ms buffer
        if optimal_chunk != self.config.get('chunk_size'):
            self.config['chunk_size'] = optimal_chunk
            fixed_issues.append(f"Optimized chunk_size to {optimal_chunk}")
        
        # Fix 5: Set reasonable gain boost
        if self.config.get('gain_boost_db', 0) > 6:
            self.config['gain_boost_db'] = 3.0
            fixed_issues.append("Reduced gain_boost_db to 3.0dB")
        
        # Save configuration if changes were made
        if fixed_issues:
            self._save_config()
            logging.info(f"Fixed {len(fixed_issues)} configuration issues:")
            for fix in fixed_issues:
                logging.info(f"  ✓ {fix}")
            return True
        else:
            logging.info("No configuration issues found to fix")
            return False
    
    def generate_report(self, analyses: List[Dict]) -> str:
        """Generate a comprehensive diagnostic report."""
        report = []
        report.append("Audio Quality Diagnostic Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        total_issues = sum(len(analysis.get('issues', [])) for analysis in analyses)
        report.append(f"Total Issues Found: {total_issues}")
        report.append("")
        
        # Configuration analysis
        if any('config_issues' in analysis for analysis in analyses):
            report.append("Configuration Issues:")
            report.append("-" * 30)
            for analysis in analyses:
                for issue in analysis.get('config_issues', []):
                    report.append(f"  • {issue}")
            report.append("")
        
        # File analyses
        file_analyses = [a for a in analyses if 'file_path' in a]
        if file_analyses:
            report.append("Audio File Analysis:")
            report.append("-" * 30)
            for analysis in file_analyses:
                file_name = Path(analysis['file_path']).name
                report.append(f"\nFile: {file_name}")
                
                stats = analysis.get('stats', {})
                if stats:
                    report.append(f"  Duration: {stats.get('duration', 0):.2f}s")
                    report.append(f"  Sample Rate: {stats.get('sample_rate', 0)}Hz")
                    report.append(f"  RMS Level: {stats.get('rms_db', 0):.1f}dB")
                
                issues = analysis.get('issues', [])
                if issues:
                    report.append("  Issues:")
                    for issue in issues:
                        report.append(f"    • {issue}")
            report.append("")
        
        # Recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.get('recommendations', []))
        
        if all_recommendations:
            report.append("Recommendations:")
            report.append("-" * 30)
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in all_recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            for i, rec in enumerate(unique_recommendations, 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)


def main():
    """Main entry point for audio diagnostics."""
    parser = argparse.ArgumentParser(description="Diagnose audio recording quality issues")
    parser.add_argument('--analyze-file', help="Analyze specific audio file")
    parser.add_argument('--test-recording', action='store_true', help="Test live recording")
    parser.add_argument('--fix-config', action='store_true', help="Automatically fix configuration issues")
    parser.add_argument('--duration', type=float, default=5.0, help="Recording test duration (seconds)")
    
    args = parser.parse_args()
    
    # Initialize diagnostics
    diagnostics = AudioDiagnostics()
    analyses = []
    
    try:
        # Configuration diagnosis
        config_analysis = diagnostics.diagnose_configuration()
        analyses.append(config_analysis)
        
        # File analysis
        if args.analyze_file:
            if os.path.exists(args.analyze_file):
                file_analysis = diagnostics.analyze_recorded_file(args.analyze_file)
                analyses.append(file_analysis)
            else:
                logging.error(f"File not found: {args.analyze_file}")
        else:
            # Analyze most recent recording
            rec_dir = project_root / 'rec'
            if rec_dir.exists():
                audio_files = list(rec_dir.glob('*.wav'))
                if audio_files:
                    most_recent = max(audio_files, key=lambda f: f.stat().st_mtime)
                    logging.info(f"Analyzing most recent recording: {most_recent.name}")
                    file_analysis = diagnostics.analyze_recorded_file(str(most_recent))
                    analyses.append(file_analysis)
        
        # Live recording test
        if args.test_recording:
            test_analysis = diagnostics.test_live_recording(args.duration)
            analyses.append(test_analysis)
        
        # Fix configuration if requested
        if args.fix_config:
            fixed = diagnostics.fix_common_issues()
            if fixed:
                logging.info("Configuration fixes applied. Please restart the whisper service.")
        
        # Generate and display report
        report = diagnostics.generate_report(analyses)
        print("\n" + report)
        
        # Save report to file
        report_file = project_root / 'log' / f'audio_diagnostic_report_{int(time.time())}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logging.info(f"Diagnostic report saved to: {report_file}")
        
    except KeyboardInterrupt:
        logging.info("Diagnostics interrupted by user")
    except Exception as e:
        logging.error(f"Diagnostics failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())