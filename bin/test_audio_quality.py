#!/home/kdresdell/Documents/DEV/whisper/venv/bin/python
"""
Quick Audio Quality Test
========================

This script tests the audio recording quality with the new simplified pipeline.
It records a short sample and analyzes the quality to verify our fixes.

Usage:
    python bin/test_audio_quality.py

Author: Claude Code - DevOps Automation Specialist
"""

import sys
import time
import json
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bin.voice_transcriber import WhisperApp
from bin.audio_diagnostics import AudioDiagnostics

def test_recording_quality():
    """Test recording quality with new settings."""
    print("🎤 Testing Audio Recording Quality")
    print("=" * 50)
    
    try:
        # Initialize the whisper app
        app = WhisperApp()
        
        print("📊 Current Configuration:")
        print(f"  Sample Rate: {app.config.get('sample_rate')} Hz")
        print(f"  Channels: {app.config.get('channels')}")
        print(f"  Chunk Size: {app.config.get('chunk_size')}")
        print(f"  Microphone Gain: {app.config.get('microphone_gain')}x")
        print(f"  Auto Gain Control: {app.config.get('auto_gain_control')}")
        print(f"  Normalization: {app.config.get('normalize_audio')}")
        print(f"  Compression: {app.config.get('compressor_enabled')}")
        print()
        
        # Start recording
        print("🔴 Starting 3-second test recording...")
        print("Please speak normally: 'Hello, this is a test recording for audio quality.'")
        
        # Start recording
        app.audio_processor.start_recording()
        
        # Record for 3 seconds
        time.sleep(3)
        
        # Stop recording
        print("⏹️  Stopping recording...")
        audio_file = app.audio_processor.stop_recording()
        
        if not audio_file:
            print("❌ Recording failed!")
            return False
        
        print(f"✅ Recording saved: {audio_file}")
        
        # Analyze the recording
        print("🔍 Analyzing recording quality...")
        diagnostics = AudioDiagnostics()
        analysis = diagnostics.analyze_recorded_file(audio_file)
        
        # Display results
        print("\n📈 Audio Quality Analysis:")
        print(f"  Duration: {analysis['stats'].get('duration', 0):.2f}s")
        print(f"  Sample Rate: {analysis['stats'].get('sample_rate')} Hz")
        print(f"  RMS Level: {analysis['stats'].get('rms_db', 0):.1f} dB")
        print(f"  Peak Level: {analysis['stats'].get('peak_db', 0):.1f} dB")
        print(f"  Noise Floor: {analysis['stats'].get('noise_floor_db', 0):.1f} dB")
        
        # Check for issues
        issues = analysis.get('issues', [])
        if issues:
            print(f"\n⚠️  Issues Found ({len(issues)}):")
            for issue in issues:
                print(f"    • {issue}")
        else:
            print("\n✅ No quality issues detected!")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for rec in recommendations:
                print(f"    • {rec}")
        
        # Process the audio (minimal processing)
        print("\n🔧 Testing audio processing...")
        processed_file = app.audio_processor.process_audio(audio_file)
        
        if processed_file and processed_file != audio_file:
            print(f"✅ Processed audio saved: {processed_file}")
            
            # Analyze processed audio
            processed_analysis = diagnostics.analyze_recorded_file(processed_file)
            print(f"  Processed Duration: {processed_analysis['stats'].get('duration', 0):.2f}s")
            print(f"  Processed RMS: {processed_analysis['stats'].get('rms_db', 0):.1f} dB")
        
        # Try transcription if whisper is available
        print("\n🎯 Testing transcription...")
        try:
            transcriber = app.transcriber
            text = transcriber.transcribe_audio(processed_file or audio_file)
            if text:
                print(f"✅ Transcription: '{text.strip()}'")
            else:
                print("❌ Transcription failed or returned empty text")
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        
        print("\n🎉 Audio quality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            if 'app' in locals():
                app.cleanup()
        except:
            pass

def main():
    """Main entry point."""
    success = test_recording_quality()
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())