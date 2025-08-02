#!/home/kdresdell/DEV/whisper/venv/bin/python
"""
Performance Monitor for French Transcription Quality

This script monitors the real-time performance of French transcription
and provides alerts when quality degrades or performance issues occur.

Usage:
    python bin/performance_monitor.py --monitor         # Real-time monitoring
    python bin/performance_monitor.py --compare         # Compare before/after
    python bin/performance_monitor.py --quality-check   # Check current quality
"""

import os
import sys
import json
import time
import logging
import argparse
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class TranscriptionMetrics:
    """Metrics for a single transcription."""
    timestamp: str
    audio_file: str
    transcribed_text: str
    processing_time: float
    model_used: str
    language: str
    text_length: int
    word_count: int
    french_quality_score: float
    language_consistency: bool
    has_errors: bool
    cpu_usage: float
    memory_usage: float

@dataclass
class QualityReport:
    """Quality assessment report."""
    period_start: str
    period_end: str
    total_transcriptions: int
    french_transcriptions: int
    average_quality_score: float
    language_mixing_rate: float
    error_rate: float
    average_processing_time: float
    performance_trend: str
    recommendations: List[str]

class FrenchPerformanceMonitor:
    """Monitor for French transcription performance and quality."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.log_dir = self.project_root / "log"
        self.config_dir = self.project_root / "config"
        self.metrics_dir = self.project_root / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        self.load_config()
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'acceptable': 0.4,
            'poor': 0.2
        }
    
    def setup_logging(self):
        """Setup logging for performance monitoring."""
        log_file = self.metrics_dir / "performance_monitor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """Load current configuration."""
        config_file = self.config_dir / "whisper_config.json"
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
            self.logger.warning("Configuration file not found")
    
    def calculate_french_quality_score(self, text: str) -> float:
        """Calculate quality score for French text (0-1)."""
        if not text.strip():
            return 0.0
        
        score = 0.0
        checks = 0
        
        # Check for French characteristics
        french_words = ['le', 'la', 'les', 'un', 'une', 'des', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles']
        french_verbs = ['est', 'sont', 'avoir', 'être', 'aller', 'faire', 'dire', 'voir', 'savoir', 'pouvoir']
        french_expressions = ['bonjour', 'merci', 'comment', 'aujourd', 'maintenant', 'peut-être', 'beaucoup']
        
        text_lower = text.lower()
        
        # French vocabulary presence (0.3 weight)
        french_word_count = sum(1 for word in french_words if word in text_lower)
        if french_word_count > 0:
            score += 0.3 * min(french_word_count / 3, 1.0)
        checks += 1
        
        # French verbs presence (0.2 weight)
        french_verb_count = sum(1 for verb in french_verbs if verb in text_lower)
        if french_verb_count > 0:
            score += 0.2 * min(french_verb_count / 2, 1.0)
        checks += 1
        
        # French expressions (0.2 weight)
        french_expr_count = sum(1 for expr in french_expressions if expr in text_lower)
        if french_expr_count > 0:
            score += 0.2 * min(french_expr_count / 2, 1.0)
        checks += 1
        
        # French accents (0.15 weight)
        french_accents = 'àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ'
        if any(c in text for c in french_accents):
            score += 0.15
        checks += 1
        
        # Proper capitalization (0.05 weight)
        if text and text[0].isupper():
            score += 0.05
        checks += 1
        
        # Proper punctuation (0.1 weight)
        if text.endswith(('.', '!', '?')):
            score += 0.1
        checks += 1
        
        return min(score, 1.0)
    
    def detect_language_mixing(self, text: str) -> bool:
        """Detect if text contains mixed languages."""
        if not text.strip():
            return False
        
        # Common English words that shouldn't appear in French text
        english_words = ['the', 'and', 'you', 'that', 'this', 'with', 'have', 'will', 'can', 'what', 'how', 'when']
        # Non-Latin characters (often indicates encoding/detection issues)
        non_latin_chars = ['해', '많은', '의', '를', '에', '는', '이', '가']
        
        text_lower = text.lower()
        
        # Check for English words in predominantly French text
        french_indicators = sum(1 for word in ['le', 'la', 'les', 'je', 'vous', 'est'] if word in text_lower)
        english_indicators = sum(1 for word in english_words if word in text_lower)
        
        # Check for non-Latin characters
        non_latin_present = any(char in text for char in non_latin_chars)
        
        return (english_indicators > 0 and french_indicators > 0) or non_latin_present
    
    def analyze_transcription_log(self, log_file: str) -> List[TranscriptionMetrics]:
        """Analyze transcription log file and extract metrics."""
        log_path = Path(log_file)
        if not log_path.exists():
            self.logger.error(f"Log file not found: {log_file}")
            return []
        
        metrics = []
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse log entries
            entries = content.split('--------------------------------------------------')
            
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                
                lines = entry.split('\n')
                if len(lines) < 2:
                    continue
                
                # Extract timestamp and filename
                header_line = lines[0]
                text_line = lines[1] if len(lines) > 1 else ""
                
                if not header_line.startswith('[') or not text_line.startswith('Text: '):
                    continue
                
                # Parse timestamp and filename
                try:
                    timestamp_end = header_line.find(']')
                    timestamp = header_line[1:timestamp_end]
                    audio_file = header_line[timestamp_end+2:].strip()
                    transcribed_text = text_line[6:].strip()  # Remove "Text: "
                    
                    if not transcribed_text:
                        continue
                    
                    # Calculate metrics
                    quality_score = self.calculate_french_quality_score(transcribed_text)
                    has_mixing = self.detect_language_mixing(transcribed_text)
                    
                    metric = TranscriptionMetrics(
                        timestamp=timestamp,
                        audio_file=audio_file,
                        transcribed_text=transcribed_text,
                        processing_time=0.0,  # Not available in log
                        model_used=self.config.get('whisper_model', 'unknown'),
                        language=self.config.get('language', 'auto'),
                        text_length=len(transcribed_text),
                        word_count=len(transcribed_text.split()),
                        french_quality_score=quality_score,
                        language_consistency=not has_mixing,
                        has_errors=has_mixing or quality_score < 0.3,
                        cpu_usage=0.0,  # Not available in log
                        memory_usage=0.0  # Not available in log
                    )
                    
                    metrics.append(metric)
                
                except Exception as e:
                    self.logger.warning(f"Error parsing log entry: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
        
        return metrics
    
    def generate_quality_report(self, metrics: List[TranscriptionMetrics], 
                               period_hours: int = 24) -> QualityReport:
        """Generate quality report from metrics."""
        if not metrics:
            return QualityReport(
                period_start="",
                period_end="",
                total_transcriptions=0,
                french_transcriptions=0,
                average_quality_score=0.0,
                language_mixing_rate=0.0,
                error_rate=0.0,
                average_processing_time=0.0,
                performance_trend="insufficient_data",
                recommendations=["No transcription data available"]
            )
        
        # Filter metrics to specified period
        now = datetime.now()
        cutoff_time = now - timedelta(hours=period_hours)
        
        recent_metrics = []
        for metric in metrics:
            try:
                metric_time = datetime.strptime(metric.timestamp, '%Y-%m-%d %H:%M:%S')
                if metric_time >= cutoff_time:
                    recent_metrics.append(metric)
            except:
                recent_metrics.append(metric)  # Include if can't parse timestamp
        
        if not recent_metrics:
            recent_metrics = metrics[-20:]  # Use last 20 if no recent ones
        
        # Calculate statistics
        total_transcriptions = len(recent_metrics)
        french_transcriptions = sum(1 for m in recent_metrics if m.french_quality_score > 0.2)
        
        quality_scores = [m.french_quality_score for m in recent_metrics]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        mixing_issues = sum(1 for m in recent_metrics if not m.language_consistency)
        mixing_rate = mixing_issues / total_transcriptions if total_transcriptions > 0 else 0.0
        
        error_count = sum(1 for m in recent_metrics if m.has_errors)
        error_rate = error_count / total_transcriptions if total_transcriptions > 0 else 0.0
        
        processing_times = [m.processing_time for m in recent_metrics if m.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Determine performance trend
        if len(recent_metrics) > 10:
            recent_half = recent_metrics[len(recent_metrics)//2:]
            older_half = recent_metrics[:len(recent_metrics)//2]
            
            recent_avg = sum(m.french_quality_score for m in recent_half) / len(recent_half)
            older_avg = sum(m.french_quality_score for m in older_half) / len(older_half)
            
            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Generate recommendations
        recommendations = []
        
        if average_quality < self.quality_thresholds['acceptable']:
            recommendations.append("Switch to 'small' or 'medium' Whisper model for better French accuracy")
        
        if mixing_rate > 0.2:
            recommendations.append("Force French language (set language: 'fr') to prevent auto-detection issues")
        
        if error_rate > 0.3:
            recommendations.append("Check audio quality settings and microphone configuration")
        
        if average_quality < self.quality_thresholds['good']:
            recommendations.append("Use French-specific initial prompt for better context")
            recommendations.append("Enable audio preprocessing (normalization, compression)")
        
        if avg_processing_time > 5.0:
            recommendations.append("Consider using GPU acceleration or smaller model for better performance")
        
        if not recommendations:
            recommendations.append("Current configuration appears optimal for French transcription")
        
        return QualityReport(
            period_start=recent_metrics[0].timestamp if recent_metrics else "",
            period_end=recent_metrics[-1].timestamp if recent_metrics else "",
            total_transcriptions=total_transcriptions,
            french_transcriptions=french_transcriptions,
            average_quality_score=average_quality,
            language_mixing_rate=mixing_rate,
            error_rate=error_rate,
            average_processing_time=avg_processing_time,
            performance_trend=trend,
            recommendations=recommendations
        )
    
    def save_metrics(self, metrics: List[TranscriptionMetrics]):
        """Save metrics to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.metrics_dir / f"transcription_metrics_{timestamp}.json"
        
        try:
            metrics_data = [asdict(metric) for metric in metrics]
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Metrics saved to: {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def print_quality_report(self, report: QualityReport):
        """Print formatted quality report."""
        print("\n" + "="*60)
        print("    FRENCH TRANSCRIPTION QUALITY REPORT")
        print("="*60)
        
        print(f"Period: {report.period_start} to {report.period_end}")
        print(f"Total Transcriptions: {report.total_transcriptions}")
        print(f"French Transcriptions: {report.french_transcriptions}")
        
        print(f"\nQuality Metrics:")
        print(f"  Average Quality Score: {report.average_quality_score:.3f}")
        
        # Quality rating
        if report.average_quality_score >= self.quality_thresholds['excellent']:
            rating = "EXCELLENT ✓"
        elif report.average_quality_score >= self.quality_thresholds['good']:
            rating = "GOOD ✓"
        elif report.average_quality_score >= self.quality_thresholds['acceptable']:
            rating = "ACCEPTABLE ⚠"
        else:
            rating = "POOR ✗"
        
        print(f"  Quality Rating: {rating}")
        print(f"  Language Mixing Rate: {report.language_mixing_rate:.1%}")
        print(f"  Error Rate: {report.error_rate:.1%}")
        print(f"  Performance Trend: {report.performance_trend.upper()}")
        
        if report.average_processing_time > 0:
            print(f"  Average Processing Time: {report.average_processing_time:.2f}s")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("="*60)
    
    def monitor_real_time(self, duration_minutes: int = 60):
        """Monitor transcription quality in real-time."""
        print(f"Starting real-time monitoring for {duration_minutes} minutes...")
        print("Watching for new transcriptions...")
        
        start_time = time.time()
        last_log_size = 0
        
        # Find current log file
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"transcriptions_{today}.txt"
        
        while time.time() - start_time < duration_minutes * 60:
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_log_size:
                    # New content detected
                    metrics = self.analyze_transcription_log(str(log_file))
                    if metrics:
                        latest_metric = metrics[-1]
                        quality = latest_metric.french_quality_score
                        
                        # Alert if quality is poor
                        if quality < self.quality_thresholds['acceptable']:
                            print(f"\n⚠ QUALITY ALERT: Score {quality:.3f} - {latest_metric.transcribed_text[:50]}...")
                        elif quality >= self.quality_thresholds['good']:
                            print(f"\n✓ Good quality: Score {quality:.3f} - {latest_metric.transcribed_text[:50]}...")
                    
                    last_log_size = current_size
            
            time.sleep(5)  # Check every 5 seconds
        
        print(f"\nMonitoring completed after {duration_minutes} minutes.")
    
    def compare_configurations(self, before_log: str, after_log: str):
        """Compare transcription quality before and after configuration changes."""
        print("Comparing transcription quality before/after configuration changes...")
        
        before_metrics = self.analyze_transcription_log(before_log)
        after_metrics = self.analyze_transcription_log(after_log)
        
        before_report = self.generate_quality_report(before_metrics)
        after_report = self.generate_quality_report(after_metrics)
        
        print("\n" + "="*60)
        print("    BEFORE/AFTER COMPARISON")
        print("="*60)
        
        print(f"{'Metric':<30} {'Before':<15} {'After':<15} {'Change'}")
        print("-" * 70)
        
        # Quality score comparison
        quality_change = after_report.average_quality_score - before_report.average_quality_score
        quality_symbol = "↑" if quality_change > 0 else "↓" if quality_change < 0 else "→"
        print(f"{'Average Quality Score':<30} {before_report.average_quality_score:<15.3f} "
              f"{after_report.average_quality_score:<15.3f} {quality_symbol} {quality_change:+.3f}")
        
        # Error rate comparison
        error_change = after_report.error_rate - before_report.error_rate
        error_symbol = "↓" if error_change < 0 else "↑" if error_change > 0 else "→"
        print(f"{'Error Rate':<30} {before_report.error_rate:<15.1%} "
              f"{after_report.error_rate:<15.1%} {error_symbol} {error_change:+.1%}")
        
        # Language mixing comparison
        mixing_change = after_report.language_mixing_rate - before_report.language_mixing_rate
        mixing_symbol = "↓" if mixing_change < 0 else "↑" if mixing_change > 0 else "→"
        print(f"{'Language Mixing Rate':<30} {before_report.language_mixing_rate:<15.1%} "
              f"{after_report.language_mixing_rate:<15.1%} {mixing_symbol} {mixing_change:+.1%}")
        
        print("-" * 70)
        
        # Overall assessment
        if quality_change > 0.1 and error_change < -0.1:
            assessment = "SIGNIFICANT IMPROVEMENT ✓✓"
        elif quality_change > 0.05:
            assessment = "IMPROVEMENT ✓"
        elif quality_change < -0.05:
            assessment = "DEGRADATION ✗"
        else:
            assessment = "NO SIGNIFICANT CHANGE →"
        
        print(f"\nOverall Assessment: {assessment}")
        print("="*60)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Monitor French transcription performance and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bin/performance_monitor.py --quality-check           # Check current quality
  python bin/performance_monitor.py --monitor 30             # Monitor for 30 minutes
  python bin/performance_monitor.py --compare before.txt after.txt  # Compare logs
        """
    )
    
    parser.add_argument('--quality-check', action='store_true',
                       help='Generate quality report for recent transcriptions')
    parser.add_argument('--monitor', type=int, metavar='MINUTES',
                       help='Monitor real-time transcription quality')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                       help='Compare quality between two log files')
    parser.add_argument('--log-file', type=str,
                       help='Specific log file to analyze')
    parser.add_argument('--period', type=int, default=24,
                       help='Analysis period in hours (default: 24)')
    
    args = parser.parse_args()
    
    monitor = FrenchPerformanceMonitor()
    
    try:
        if args.quality_check:
            # Find most recent log file
            log_file = args.log_file
            if not log_file:
                today = datetime.now().strftime('%Y-%m-%d')
                log_file = f"log/transcriptions_{today}.txt"
            
            metrics = monitor.analyze_transcription_log(log_file)
            if metrics:
                report = monitor.generate_quality_report(metrics, args.period)
                monitor.print_quality_report(report)
                monitor.save_metrics(metrics)
            else:
                print("No transcription data found to analyze.")
        
        elif args.monitor:
            monitor.monitor_real_time(args.monitor)
        
        elif args.compare:
            before_log, after_log = args.compare
            monitor.compare_configurations(before_log, after_log)
        
        else:
            # Default: quality check
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = f"log/transcriptions_{today}.txt"
            
            metrics = monitor.analyze_transcription_log(log_file)
            if metrics:
                report = monitor.generate_quality_report(metrics, args.period)
                monitor.print_quality_report(report)
            else:
                print("No transcription data found. Run with --help for usage options.")
    
    except KeyboardInterrupt:
        print("\n✗ Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Monitoring failed: {e}")
        logging.error(f"Monitor error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()