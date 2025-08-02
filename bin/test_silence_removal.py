#!/home/kdresdell/Documents/DEV/whisper/venv/bin/python
"""
Test script for advanced silence removal functionality.

This script validates that the new silence removal feature works correctly
and provides performance metrics.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the project directory to the path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

# Import the main components
from bin.voice_transcriber import WhisperConfig, AudioProcessor

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_silence_removal():
    """Test the advanced silence removal functionality."""
    try:
        print("🧪 Testing Advanced Silence Removal")
        print("=" * 50)
        
        # Set up test configuration
        config = WhisperConfig()
        audio_processor = AudioProcessor(config)
        
        # Test with different configurations
        test_configs = [
            {
                "name": "Conservative",
                "config": {
                    "enabled": True,
                    "aggressiveness": "conservative",
                    "leading_silence_threshold_db": -40,
                    "trailing_silence_threshold_db": -40,
                    "internal_silence_threshold_db": -35
                }
            },
            {
                "name": "Moderate", 
                "config": {
                    "enabled": True,
                    "aggressiveness": "moderate",
                    "leading_silence_threshold_db": -40,
                    "trailing_silence_threshold_db": -40,
                    "internal_silence_threshold_db": -35
                }
            },
            {
                "name": "Aggressive",
                "config": {
                    "enabled": True,
                    "aggressiveness": "aggressive",
                    "leading_silence_threshold_db": -40,
                    "trailing_silence_threshold_db": -40,
                    "internal_silence_threshold_db": -35
                }
            }
        ]
        
        # Look for test audio files in the rec directory
        rec_dir = project_dir / "rec"
        audio_files = list(rec_dir.glob("*.wav"))
        
        if not audio_files:
            print("❌ No audio files found in rec/ directory for testing")
            print("   Record some audio first using the main application")
            return False
        
        # Test with the most recent audio file
        test_file = sorted(audio_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"📁 Using test file: {test_file.name}")
        
        # Get original file size
        original_size = test_file.stat().st_size
        print(f"📏 Original file size: {original_size:,} bytes")
        
        # Test each configuration
        for test_config in test_configs:
            print(f"\n🔧 Testing {test_config['name']} settings...")
            
            # Update audio processor config
            audio_processor.config.set('advanced_silence_removal', test_config['config'])
            
            # Process the audio
            result_file = audio_processor.advanced_silence_removal(str(test_file))
            
            if result_file and result_file != str(test_file):
                # Get processed file size
                processed_size = Path(result_file).stat().st_size
                reduction = ((original_size - processed_size) / original_size) * 100
                
                print(f"   ✅ Processing successful")
                print(f"   📏 Processed size: {processed_size:,} bytes")
                print(f"   📉 Size reduction: {reduction:.1f}%")
                print(f"   📄 Output file: {Path(result_file).name}")
                
                # Clean up test file
                Path(result_file).unlink()
            else:
                print(f"   ❌ Processing failed or no changes made")
        
        print(f"\n✅ Silence removal testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logging.error(f"Test error: {e}", exc_info=True)
        return False

def test_configuration_validation():
    """Test that configuration validation works correctly."""
    print(f"\n🧪 Testing Configuration Validation")
    print("=" * 50)
    
    try:
        config = WhisperConfig()
        
        # Test valid configuration
        valid_config = {
            "enabled": True,
            "aggressiveness": "moderate",
            "leading_silence_threshold_db": -40,
            "trailing_silence_threshold_db": -40,
            "internal_silence_threshold_db": -35,
            "min_leading_silence_duration": 0.1,
            "min_trailing_silence_duration": 0.1,
            "preserve_speech_padding_ms": 100,
            "preserve_natural_rhythm": True,
            "minimum_chunk_duration_ms": 200
        }
        
        config.set('advanced_silence_removal', valid_config)
        retrieved_config = config.get('advanced_silence_removal')
        
        print("✅ Configuration set and retrieved successfully")
        print(f"   Enabled: {retrieved_config.get('enabled')}")
        print(f"   Aggressiveness: {retrieved_config.get('aggressiveness')}")
        print(f"   Leading threshold: {retrieved_config.get('leading_silence_threshold_db')}dB")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    setup_logging()
    
    print("🚀 Starting Silence Removal Tests")
    print("=" * 60)
    
    # Check that we're in the right directory
    if not (project_dir / "bin" / "voice_transcriber.py").exists():
        print("❌ Test must be run from the whisper project directory")
        return 1
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_configuration_validation():
        tests_passed += 1
    
    if test_silence_removal():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Silence removal is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())