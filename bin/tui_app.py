#!/usr/bin/env python3
"""
TUI Dashboard for Voice Transcription Application

A beautiful terminal user interface with:
- Real-time audio level visualization
- Transcription output display with copy/paste
- Recording file manager
- Backend switching and settings

Author: Python DevOps Automation Specialist
Compatible: Arch Linux, Wayland/Sway
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Label, Button, DataTable,
    TextArea, ProgressBar, Select, Log, RichLog
)
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.progress_bar import ProgressBar as RichProgressBar
from rich.table import Table as RichTable
from rich import box


# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'bin'))


class AudioLevelPanel(Static):
    """Panel showing real-time audio levels with animated meter."""

    current_level = reactive(0.0)
    peak_level = reactive(0.0)
    is_recording = reactive(False)
    recording_status = reactive("IDLE")
    gain = reactive(1.0)

    def __init__(self):
        super().__init__()
        self.border_title = "Audio Levels"

    def compose(self) -> ComposeResult:
        """Create audio level display."""
        yield Static(id="audio-meter")
        yield Static(id="audio-stats")

    def watch_current_level(self, level: float) -> None:
        """Update display when level changes."""
        self.update_meter()

    def watch_is_recording(self, recording: bool) -> None:
        """Update display when recording status changes."""
        self.update_meter()

    def update_meter(self) -> None:
        """Update the audio meter visualization."""
        # Create level bar (0-100 scale)
        level_pct = min(int(self.current_level * 100), 100)
        peak_pct = min(int(self.peak_level * 100), 100)

        # Calculate dB
        db_level = -60 if self.current_level <= 0 else 20 * (self.current_level ** 0.5) - 60

        # Create visual meter with color zones
        meter_width = 50
        filled = int((level_pct / 100) * meter_width)
        peak_pos = int((peak_pct / 100) * meter_width)

        bar = []
        for i in range(meter_width):
            if i < filled:
                if i < meter_width * 0.6:  # Green zone
                    bar.append('[green]█[/green]')
                elif i < meter_width * 0.8:  # Yellow zone
                    bar.append('[yellow]█[/yellow]')
                else:  # Red zone
                    bar.append('[red]█[/red]')
            elif i == peak_pos and i >= filled:
                bar.append('[bold white]|[/bold white]')
            else:
                bar.append('[dim]░[/dim]')

        meter_display = ''.join(bar)

        # Recording status indicator
        if self.is_recording:
            status_indicator = "[bold red]● RECORDING[/bold red]"
        else:
            status_indicator = "[dim]○ {status}[/dim]".format(status=self.recording_status)

        # Build meter panel
        meter_widget = self.query_one("#audio-meter", Static)
        meter_widget.update(
            f"{status_indicator}\n\n"
            f"[bold]Level:[/bold] [{meter_display}] {db_level:+.1f} dB\n"
            f"[dim]Peak: {peak_pct}% | Gain: {self.gain:.2f}x[/dim]"
        )

    def set_level(self, current: float, peak: float, gain: float = 1.0) -> None:
        """Set audio levels.

        Args:
            current: Current RMS level (0.0-1.0)
            peak: Peak level (0.0-1.0)
            gain: Current gain multiplier
        """
        self.current_level = current
        self.peak_level = peak
        self.gain = gain

    def set_recording_status(self, recording: bool, status: str = "IDLE") -> None:
        """Set recording status.

        Args:
            recording: True if currently recording
            status: Status text (IDLE, RECORDING, PROCESSING, etc.)
        """
        self.is_recording = recording
        self.recording_status = status


class TranscriptionPanel(Static):
    """Panel showing transcription results with history."""

    def __init__(self):
        super().__init__()
        self.border_title = "Transcriptions"
        self.transcriptions = []

    def compose(self) -> ComposeResult:
        """Create transcription display."""
        with Vertical():
            yield Static("[bold]Current Transcription:[/bold]", id="current-label")
            yield TextArea("", id="current-transcription", read_only=True)
            yield Horizontal(
                Button("Copy", id="copy-btn", variant="primary"),
                Button("Clear", id="clear-btn"),
                Button("Export", id="export-btn"),
                classes="button-row"
            )
            yield Static("[bold]Recent History:[/bold]", classes="history-label")
            yield RichLog(id="transcription-history", highlight=True, markup=True)

    def add_transcription(self, text: str, timestamp: Optional[datetime] = None) -> None:
        """Add a new transcription.

        Args:
            text: Transcribed text
            timestamp: Timestamp of transcription (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Update current transcription
        current_area = self.query_one("#current-transcription", TextArea)
        current_area.text = text

        # Add to history
        self.transcriptions.insert(0, {'text': text, 'timestamp': timestamp})
        self.transcriptions = self.transcriptions[:20]  # Keep last 20

        # Update history display
        history_log = self.query_one("#transcription-history", RichLog)
        history_log.clear()
        for item in self.transcriptions:
            time_str = item['timestamp'].strftime("%H:%M:%S")
            preview = item['text'][:80] + "..." if len(item['text']) > 80 else item['text']
            history_log.write(f"[dim]{time_str}[/dim] {preview}")

    def get_current_text(self) -> str:
        """Get current transcription text."""
        current_area = self.query_one("#current-transcription", TextArea)
        return current_area.text

    def clear_current(self) -> None:
        """Clear current transcription."""
        current_area = self.query_one("#current-transcription", TextArea)
        current_area.text = ""


class FileManagerPanel(Static):
    """Panel for managing recording files."""

    def __init__(self, recordings_dir: str = "rec"):
        super().__init__()
        self.border_title = "Recording Files"
        self.recordings_dir = Path(recordings_dir)
        self.selected_files = set()

    def compose(self) -> ComposeResult:
        """Create file manager display."""
        with Vertical():
            yield Static(f"[dim]Directory: {self.recordings_dir}[/dim]", id="dir-label")
            yield DataTable(id="files-table")
            yield Horizontal(
                Button("Delete Selected", id="delete-btn", variant="error"),
                Button("Play", id="play-btn"),
                Button("Refresh", id="refresh-btn"),
                Button("Clean Old (7d+)", id="cleanup-btn"),
                classes="button-row"
            )
            yield Static("", id="storage-info")

    def on_mount(self) -> None:
        """Setup file table on mount."""
        table = self.query_one("#files-table", DataTable)
        table.add_columns("File", "Date", "Size", "Duration")
        table.cursor_type = "row"
        self.refresh_files()

    def refresh_files(self) -> None:
        """Refresh the file list."""
        table = self.query_one("#files-table", DataTable)
        table.clear()

        if not self.recordings_dir.exists():
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            return

        files = sorted(
            self.recordings_dir.glob("*.wav"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        total_size = 0
        for file_path in files:
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            total_size += stat.st_size
            date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            # Estimate duration from file size (rough estimate)
            duration = self._estimate_duration(file_path)

            table.add_row(
                file_path.name,
                date_str,
                f"{size_mb:.2f} MB",
                duration
            )

        # Update storage info
        storage_widget = self.query_one("#storage-info", Static)
        total_mb = total_size / (1024 * 1024)
        file_count = len(files)
        storage_widget.update(
            f"[dim]Total: {file_count} files, {total_mb:.2f} MB[/dim]"
        )

    def _estimate_duration(self, file_path: Path) -> str:
        """Estimate audio duration from file.

        Args:
            file_path: Path to WAV file

        Returns:
            str: Duration string (e.g., "00:15")
        """
        try:
            import wave
            with wave.open(str(file_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                mins = int(duration // 60)
                secs = int(duration % 60)
                return f"{mins:02d}:{secs:02d}"
        except Exception:
            return "??:??"

    def get_selected_files(self) -> List[Path]:
        """Get list of selected files.

        Returns:
            list: Selected file paths
        """
        table = self.query_one("#files-table", DataTable)
        selected = []

        if table.cursor_row is not None:
            row = table.get_row_at(table.cursor_row)
            filename = row[0]
            selected.append(self.recordings_dir / filename)

        return selected

    def delete_selected(self) -> int:
        """Delete selected files.

        Returns:
            int: Number of files deleted
        """
        files = self.get_selected_files()
        deleted = 0

        for file_path in files:
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                logging.error(f"Error deleting {file_path}: {e}")

        self.refresh_files()
        return deleted


class SettingsSidebar(Static):
    """Sidebar with backend selection and settings."""

    def __init__(self):
        super().__init__()
        self.border_title = "Settings"

    def compose(self) -> ComposeResult:
        """Create settings sidebar."""
        with Vertical():
            yield Static("[bold]Backend[/bold]")
            yield Select(
                [("Whisper", "whisper"), ("Vosk", "vosk")],
                id="backend-select",
                value="vosk"
            )
            yield Static("\n[bold]Status[/bold]")
            yield Static("Backend: [green]Loaded[/green]", id="backend-status")
            yield Static("Model: base", id="model-info")
            yield Static("\n[bold]Performance[/bold]", classes="section-label")
            yield Static("CPU: 0%", id="cpu-usage")
            yield Static("RAM: 0 MB", id="ram-usage")
            yield Static("Last: 0.0s", id="last-transcription-time")
            yield Static("\n")
            yield Button("Settings", id="settings-btn", variant="default")
            yield Button("Quit (Q)", id="quit-btn", variant="error")

    def update_backend_status(self, backend: str, model: str, loaded: bool) -> None:
        """Update backend status display."""
        status_widget = self.query_one("#backend-status", Static)
        model_widget = self.query_one("#model-info", Static)

        status_color = "green" if loaded else "red"
        status_text = "Loaded" if loaded else "Not Loaded"

        status_widget.update(f"Backend: [{status_color}]{status_text}[/{status_color}]")
        model_widget.update(f"Model: {model}")

    def update_performance(self, cpu: float, ram: float, last_time: float) -> None:
        """Update performance metrics."""
        self.query_one("#cpu-usage", Static).update(f"CPU: {cpu:.1f}%")
        self.query_one("#ram-usage", Static).update(f"RAM: {ram:.0f} MB")
        self.query_one("#last-transcription-time", Static).update(f"Last: {last_time:.1f}s")


class VoiceTranscriberTUI(App):
    """Main TUI application for voice transcription."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 4 3;
        grid-rows: 8 1fr 10;
        grid-columns: 1fr 25;
    }

    AudioLevelPanel {
        column-span: 3;
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    SettingsSidebar {
        row-span: 3;
        border: solid $accent;
        padding: 1;
    }

    TranscriptionPanel {
        column-span: 3;
        border: solid $success;
        padding: 1;
        height: 100%;
    }

    FileManagerPanel {
        column-span: 3;
        border: solid $warning;
        padding: 1;
        height: 100%;
    }

    .button-row {
        height: auto;
        padding: 1;
    }

    #current-transcription {
        height: 10;
        border: solid $primary-lighten-1;
    }

    #transcription-history {
        height: 100%;
        border: solid $primary-darken-1;
    }

    #files-table {
        height: 100%;
    }

    .section-label {
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "toggle_recording", "Record", show=True),
        Binding("c", "copy_text", "Copy", show=True),
        Binding("d", "delete_file", "Delete", show=True),
        Binding("b", "switch_backend", "Backend", show=True),
    ]

    def __init__(self, config: Dict[str, Any]):
        """Initialize TUI app.

        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.title = "Voice Transcriber"
        self.sub_title = "Press R to record, Q to quit"

        # State
        self.is_recording = False
        self.backend = None
        self.audio_processor = None
        self.performance_thread = None

    def compose(self) -> ComposeResult:
        """Create the TUI layout."""
        yield Header()
        yield AudioLevelPanel()
        yield TranscriptionPanel()
        yield FileManagerPanel(self.config.get('recordings_dir', 'rec'))
        yield SettingsSidebar()
        yield Footer()

    def on_mount(self) -> None:
        """Setup when app is mounted."""
        # Initialize backend
        self.load_backend()

        # Start performance monitoring
        self.start_performance_monitoring()

    def load_backend(self) -> None:
        """Load the transcription backend."""
        from transcription_backends import BackendFactory

        try:
            self.backend = BackendFactory.create_backend(self.config)
            success = self.backend.load_model()

            sidebar = self.query_one(SettingsSidebar)
            info = self.backend.get_model_info()
            sidebar.update_backend_status(
                info['backend'],
                info.get('model', 'unknown'),
                success
            )

            if success:
                self.notify(f"{info['backend']} backend loaded successfully", severity="information")
            else:
                self.notify(f"Failed to load {info['backend']} backend", severity="error")

        except Exception as e:
            self.notify(f"Error loading backend: {e}", severity="error")
            logging.error(f"Backend load error: {e}")

    def action_toggle_recording(self) -> None:
        """Toggle recording state."""
        audio_panel = self.query_one(AudioLevelPanel)

        if not self.is_recording:
            # Start recording
            self.is_recording = True
            audio_panel.set_recording_status(True, "RECORDING")
            self.notify("Recording started...", severity="information")
            # TODO: Actually start recording via audio processor
        else:
            # Stop recording and transcribe
            self.is_recording = False
            audio_panel.set_recording_status(False, "PROCESSING")
            self.notify("Processing transcription...", severity="information")
            # TODO: Stop recording and process
            # Simulate transcription for now
            self.simulate_transcription()

    def simulate_transcription(self) -> None:
        """Simulate a transcription (for testing)."""
        # This will be replaced with actual transcription
        transcription_panel = self.query_one(TranscriptionPanel)
        audio_panel = self.query_one(AudioLevelPanel)

        # Simulate processing delay
        import time
        time.sleep(0.5)

        sample_text = f"This is a simulated transcription at {datetime.now().strftime('%H:%M:%S')}"
        transcription_panel.add_transcription(sample_text)

        audio_panel.set_recording_status(False, "IDLE")
        self.notify("Transcription complete!", severity="success")

    def action_copy_text(self) -> None:
        """Copy current transcription to clipboard."""
        transcription_panel = self.query_one(TranscriptionPanel)
        text = transcription_panel.get_current_text()

        if text:
            try:
                import pyperclip
                pyperclip.copy(text)
                self.notify("Copied to clipboard!", severity="success")
            except Exception as e:
                self.notify(f"Copy failed: {e}", severity="error")
        else:
            self.notify("No text to copy", severity="warning")

    def action_delete_file(self) -> None:
        """Delete selected recording file."""
        file_panel = self.query_one(FileManagerPanel)
        deleted = file_panel.delete_selected()

        if deleted > 0:
            self.notify(f"Deleted {deleted} file(s)", severity="success")
        else:
            self.notify("No file selected", severity="warning")

    def action_switch_backend(self) -> None:
        """Switch between backends."""
        current = self.config.get('backend', 'vosk')
        new_backend = 'whisper' if current == 'vosk' else 'vosk'

        self.config['backend'] = new_backend
        self.notify(f"Switching to {new_backend}...", severity="information")

        # Reload backend
        if self.backend:
            self.backend.unload_model()
        self.load_backend()

    def start_performance_monitoring(self) -> None:
        """Start background performance monitoring."""
        def monitor():
            import psutil
            import time

            process = psutil.Process()
            sidebar = self.query_one(SettingsSidebar)

            while True:
                try:
                    cpu = process.cpu_percent()
                    ram = process.memory_info().rss / (1024 * 1024)  # MB
                    last_time = 0.0  # TODO: Track actual transcription time

                    # Update sidebar (must be done on main thread)
                    self.call_from_thread(sidebar.update_performance, cpu, ram, last_time)

                    time.sleep(1.0)
                except Exception as e:
                    logging.error(f"Performance monitoring error: {e}")
                    break

        self.performance_thread = threading.Thread(target=monitor, daemon=True)
        self.performance_thread.start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "copy-btn":
            self.action_copy_text()
        elif button_id == "clear-btn":
            transcription_panel = self.query_one(TranscriptionPanel)
            transcription_panel.clear_current()
            self.notify("Cleared", severity="information")
        elif button_id == "delete-btn":
            self.action_delete_file()
        elif button_id == "refresh-btn":
            file_panel = self.query_one(FileManagerPanel)
            file_panel.refresh_files()
            self.notify("Refreshed", severity="information")
        elif button_id == "quit-btn":
            self.action_quit()


def run_tui(config: Dict[str, Any]) -> None:
    """Run the TUI application.

    Args:
        config: Application configuration dictionary
    """
    app = VoiceTranscriberTUI(config)
    app.run()


if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)

    test_config = {
        'backend': 'vosk',
        'vosk_model_path': './models/vosk-model-small-en-us-0.15',
        'recordings_dir': 'rec',
        'language': 'en'
    }

    run_tui(test_config)
