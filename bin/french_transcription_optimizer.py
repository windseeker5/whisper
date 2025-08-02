#!/home/kdresdell/DEV/whisper/venv/bin/python
"""
French Transcription Optimizer for Whisper Voice-to-Text Application

This script provides tools to optimize French transcription quality by:
1. Testing different Whisper model sizes for French accuracy
2. Benchmarking performance vs quality trade-offs
3. Optimizing audio preprocessing for French phonetics
4. Comparing transcription results with different configurations

Usage:
    python bin/french_transcription_optimizer.py --test-models
    python bin/french_transcription_optimizer.py --benchmark-audio
    python bin/french_transcription_optimizer.py --analyze-logs
    python bin/french_transcription_optimizer.py --optimize-config
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import whisper
    import numpy as np
    from scipy.io import wavfile
    import librosa
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install: {e}")
    print("Run: pip install openai-whisper numpy scipy librosa")
    sys.exit(1)

@dataclass
class ModelPerformance:
    """Performance metrics for a Whisper model configuration."""
    model_name: str
    language: str
    load_time: float
    transcription_time: float
    memory_usage: float
    accuracy_score: float
    word_error_rate: float
    text_quality: str

@dataclass
class FrenchTestResult:
    """Results from French transcription testing."""
    config: Dict[str, Any]
    performance: ModelPerformance
    transcribed_text: str
    confidence_score: float
    language_consistency: float

class FrenchTranscriptionOptimizer:
    """Optimizer for French transcription quality and performance."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.test_audio_dir = Path("rec")
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Whisper models to test (ordered by size)
        self.models_to_test = ["tiny", "base", "small", "medium"]
        
        # French test phrases for quality assessment
        self.french_test_phrases = [
            "Bonjour, comment allez-vous aujourd'hui ?",
            "Je voudrais commander un café s'il vous plaît.",
            "Quelle heure est-il maintenant ?",
            "Pouvez-vous m'aider avec cette transcription ?",
            "La qualité de la reconnaissance vocale est importante.",
            "Les accents français peuvent être difficiles à transcrire.",
            "Merci beaucoup pour votre assistance technique.",
            "Au revoir et bonne journée à vous."
        ]
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for optimization process."""
        log_file = self.results_dir / "french_optimization.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_current_config(self) -> Dict[str, Any]:
        """Load current whisper configuration."""
        config_file = self.config_dir / "whisper_config.json"
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_file}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            return {}
    
    def save_optimized_config(self, config: Dict[str, Any], suffix: str = "_optimized"):
        """Save optimized configuration."""
        config_file = self.config_dir / f"whisper_config{suffix}.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved optimized config to: {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def test_model_performance(self, model_name: str, test_audio: str) -> ModelPerformance:
        """Test performance metrics for a specific Whisper model."""
        self.logger.info(f"Testing model: {model_name}")
        
        # Load model and measure load time
        start_time = time.time()
        try:
            model = whisper.load_model(model_name)
            load_time = time.time() - start_time
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
        
        # Test transcription performance
        transcription_start = time.time()
        try:
            result = model.transcribe(
                test_audio,
                language='fr',
                beam_size=10,
                best_of=10,
                temperature=(0.0, 0.1, 0.2, 0.3, 0.5),
                condition_on_previous_text=True,
                initial_prompt="Transcription en français. Bonjour, comment allez-vous ?"
            )
            transcription_time = time.time() - transcription_start
            transcribed_text = result['text'].strip()
        except Exception as e:
            self.logger.error(f"Transcription failed for model {model_name}: {e}")
            return None
        
        # Calculate basic metrics
        memory_usage = self.estimate_memory_usage(model)
        
        return ModelPerformance(
            model_name=model_name,
            language='fr',
            load_time=load_time,
            transcription_time=transcription_time,
            memory_usage=memory_usage,
            accuracy_score=0.0,  # Would need reference text for real calculation
            word_error_rate=0.0,  # Would need reference text for real calculation
            text_quality=transcribed_text
        )
    
    def estimate_memory_usage(self, model) -> float:
        """Estimate memory usage of a Whisper model (simplified)."""
        try:
            # Count parameters (rough estimate)
            total_params = sum(p.numel() for p in model.parameters())
            # Estimate memory in MB (4 bytes per float32 parameter)
            memory_mb = (total_params * 4) / (1024 * 1024)
            return memory_mb
        except:
            return 0.0
    
    def analyze_french_quality(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze the quality of French transcription."""
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_french_accents': any(c in text for c in 'àâäéèêëïîôöùûüÿç'),
            'has_french_articles': any(word in text.lower() for word in ['le', 'la', 'les', 'un', 'une', 'des']),
            'has_french_verbs': any(word in text.lower() for word in ['est', 'sont', 'avoir', 'être', 'aller']),
            'capitalization_proper': text[0].isupper() if text else False,
            'ends_with_punctuation': text.endswith(('.', '!', '?')) if text else False
        }
        
        # Calculate quality score (0-1)
        score = 0.0
        if analysis['has_french_accents']:
            score += 0.2
        if analysis['has_french_articles']:
            score += 0.2
        if analysis['has_french_verbs']:
            score += 0.2
        if analysis['capitalization_proper']:
            score += 0.1
        if analysis['ends_with_punctuation']:
            score += 0.1
        if analysis['word_count'] > 2:
            score += 0.2
        
        return score, analysis
    
    def benchmark_all_models(self) -> List[ModelPerformance]:
        """Benchmark all available Whisper models for French."""
        results = []
        
        # Find a test audio file
        test_files = list(self.test_audio_dir.glob("*.wav"))
        if not test_files:
            self.logger.error("No test audio files found in rec/ directory")
            return results
        
        test_audio = str(test_files[0])
        self.logger.info(f"Using test audio: {test_audio}")
        
        for model_name in self.models_to_test:
            try:
                performance = self.test_model_performance(model_name, test_audio)
                if performance:
                    results.append(performance)
                    self.logger.info(f"Model {model_name}: "
                                   f"Load: {performance.load_time:.2f}s, "
                                   f"Transcribe: {performance.transcription_time:.2f}s, "
                                   f"Memory: {performance.memory_usage:.1f}MB")
            except Exception as e:
                self.logger.error(f"Failed to test model {model_name}: {e}")
        
        return results
    
    def generate_optimization_report(self, results: List[ModelPerformance]) -> str:
        """Generate a comprehensive optimization report."""
        report = []
        report.append("=== French Transcription Optimization Report ===\n")
        
        if not results:
            report.append("No model test results available.\n")
            return "\n".join(report)
        
        # Performance comparison table
        report.append("Model Performance Comparison:")
        report.append("-" * 80)
        report.append(f"{'Model':<10} {'Load Time':<12} {'Transcribe':<12} {'Memory':<10} {'Quality Preview'}")
        report.append("-" * 80)
        
        for result in results:
            quality_preview = result.text_quality[:30] + "..." if len(result.text_quality) > 30 else result.text_quality
            report.append(f"{result.model_name:<10} {result.load_time:<12.2f} "
                         f"{result.transcription_time:<12.2f} {result.memory_usage:<10.1f} {quality_preview}")
        
        report.append("-" * 80)
        
        # Recommendations
        report.append("\nRecommendations:")
        
        # Find best performance/quality balance
        if len(results) >= 2:
            fastest = min(results, key=lambda x: x.transcription_time)
            smallest_memory = min(results, key=lambda x: x.memory_usage)
            
            report.append(f"• Fastest transcription: {fastest.model_name} ({fastest.transcription_time:.2f}s)")
            report.append(f"• Lowest memory usage: {smallest_memory.model_name} ({smallest_memory.memory_usage:.1f}MB)")
            
            # Recommend based on balance
            if any(r.model_name == 'small' for r in results):
                report.append("• RECOMMENDED for French: 'small' model (best balance of speed/accuracy)")
            elif any(r.model_name == 'base' for r in results):
                report.append("• RECOMMENDED for French: 'base' model (current choice, upgrade to 'small' if possible)")
        
        # Configuration recommendations
        report.append("\nOptimal Configuration for French:")
        optimal_config = self.generate_optimal_french_config()
        for key, value in optimal_config.items():
            report.append(f"• {key}: {value}")
        
        return "\n".join(report)
    
    def generate_optimal_french_config(self) -> Dict[str, Any]:
        """Generate optimal configuration for French transcription."""
        return {
            "whisper_model": "small",
            "language": "fr",
            "sample_rate": 44100,
            "channels": 1,
            "microphone_gain": 1.2,
            "auto_gain_control": True,
            "target_rms_level": 0.15,
            "gain_boost_db": 3.0,
            "noise_gate_threshold": -35.0,
            "compressor_enabled": True,
            "normalize_audio": True,
            "vad_threshold": 0.01,
            "silence_threshold": -30,
            "min_silence_len": 500
        }
    
    def analyze_existing_logs(self, log_file: str = None) -> Dict[str, Any]:
        """Analyze existing transcription logs for French quality issues."""
        if not log_file:
            log_file = "log/transcriptions_2025-08-02.txt"
        
        log_path = Path(log_file)
        if not log_path.exists():
            self.logger.error(f"Log file not found: {log_path}")
            return {}
        
        analysis = {
            'total_transcriptions': 0,
            'french_transcriptions': 0,
            'mixed_language_issues': 0,
            'quality_issues': [],
            'recommendations': []
        }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into individual transcriptions
            transcriptions = []
            for line in content.split('\n'):
                if line.startswith('Text: '):
                    text = line[6:]  # Remove "Text: " prefix
                    transcriptions.append(text)
            
            analysis['total_transcriptions'] = len(transcriptions)
            
            for text in transcriptions:
                if not text.strip():
                    continue
                
                # Check for French content
                has_french = any(word in text.lower() for word in 
                               ['bonjour', 'merci', 'comment', 'aujourd', 'vous', 'est', 'sont', 'le', 'la', 'les'])
                
                if has_french:
                    analysis['french_transcriptions'] += 1
                    
                    # Check for quality issues
                    if any(char in text for char in ['해', '많은']):  # Non-French characters
                        analysis['mixed_language_issues'] += 1
                        analysis['quality_issues'].append(f"Mixed languages: {text[:50]}...")
                    
                    # Check for French-specific issues
                    quality_score, quality_analysis = self.analyze_french_quality(text)
                    if quality_score < 0.3:
                        analysis['quality_issues'].append(f"Low quality French: {text[:50]}...")
            
            # Generate recommendations
            if analysis['mixed_language_issues'] > 0:
                analysis['recommendations'].append("Enable language forcing (set language: 'fr')")
            
            if analysis['french_transcriptions'] > 0:
                french_ratio = analysis['french_transcriptions'] / analysis['total_transcriptions']
                if french_ratio > 0.5:
                    analysis['recommendations'].append("Switch to 'small' model for better French support")
                    analysis['recommendations'].append("Use French-specific initial prompt")
            
        except Exception as e:
            self.logger.error(f"Error analyzing logs: {e}")
        
        return analysis
    
    def run_comprehensive_optimization(self):
        """Run complete French transcription optimization."""
        self.logger.info("Starting comprehensive French transcription optimization...")
        
        # Step 1: Analyze current logs
        self.logger.info("Step 1: Analyzing existing transcription logs...")
        log_analysis = self.analyze_existing_logs()
        
        # Step 2: Benchmark models
        self.logger.info("Step 2: Benchmarking Whisper models for French...")
        model_results = self.benchmark_all_models()
        
        # Step 3: Generate optimization report
        self.logger.info("Step 3: Generating optimization report...")
        report = self.generate_optimization_report(model_results)
        
        # Step 4: Save optimized configuration
        self.logger.info("Step 4: Creating optimized configuration...")
        optimal_config = self.load_current_config()
        optimal_config.update(self.generate_optimal_french_config())
        self.save_optimized_config(optimal_config)
        
        # Save report
        report_file = self.results_dir / "french_optimization_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write("\n\n=== Log Analysis Results ===\n")
            f.write(f"Total transcriptions analyzed: {log_analysis.get('total_transcriptions', 0)}\n")
            f.write(f"French transcriptions detected: {log_analysis.get('french_transcriptions', 0)}\n")
            f.write(f"Mixed language issues: {log_analysis.get('mixed_language_issues', 0)}\n")
            
            if log_analysis.get('quality_issues'):
                f.write("\nQuality Issues Found:\n")
                for issue in log_analysis['quality_issues'][:5]:  # Show first 5 issues
                    f.write(f"- {issue}\n")
            
            if log_analysis.get('recommendations'):
                f.write("\nLog-based Recommendations:\n")
                for rec in log_analysis['recommendations']:
                    f.write(f"- {rec}\n")
        
        self.logger.info(f"Optimization complete! Report saved to: {report_file}")
        return report_file

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Optimize French transcription quality for Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/french_transcription_optimizer.py --optimize     # Run full optimization
  python bin/french_transcription_optimizer.py --test-models # Test model performance
  python bin/french_transcription_optimizer.py --analyze     # Analyze existing logs
  python bin/french_transcription_optimizer.py --config      # Generate optimal config
        """
    )
    
    parser.add_argument('--optimize', action='store_true',
                       help='Run comprehensive French optimization')
    parser.add_argument('--test-models', action='store_true',
                       help='Test different Whisper models for French')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing transcription logs')
    parser.add_argument('--config', action='store_true',
                       help='Generate optimal French configuration')
    parser.add_argument('--log-file', type=str,
                       help='Specific log file to analyze')
    
    args = parser.parse_args()
    
    # If no arguments provided, run full optimization
    if not any([args.optimize, args.test_models, args.analyze, args.config]):
        args.optimize = True
    
    optimizer = FrenchTranscriptionOptimizer()
    
    try:
        if args.optimize:
            report_file = optimizer.run_comprehensive_optimization()
            print(f"\n✓ French transcription optimization completed!")
            print(f"✓ Report saved to: {report_file}")
            print(f"✓ Optimized configuration created")
            print(f"\nTo apply the optimization:")
            print(f"1. Review the report: cat {report_file}")
            print(f"2. Backup current config: cp config/whisper_config.json config/whisper_config_backup.json")
            print(f"3. Apply optimized config: cp config/whisper_config_optimized.json config/whisper_config.json")
            print(f"4. Restart the whisper service if running")
        
        elif args.test_models:
            results = optimizer.benchmark_all_models()
            report = optimizer.generate_optimization_report(results)
            print(report)
        
        elif args.analyze:
            analysis = optimizer.analyze_existing_logs(args.log_file)
            print(f"Log Analysis Results:")
            print(f"Total transcriptions: {analysis.get('total_transcriptions', 0)}")
            print(f"French transcriptions: {analysis.get('french_transcriptions', 0)}")
            print(f"Mixed language issues: {analysis.get('mixed_language_issues', 0)}")
        
        elif args.config:
            config = optimizer.generate_optimal_french_config()
            print("Optimal French Configuration:")
            print(json.dumps(config, indent=2, ensure_ascii=False))
    
    except KeyboardInterrupt:
        print("\n✗ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        logging.error(f"Optimization error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()