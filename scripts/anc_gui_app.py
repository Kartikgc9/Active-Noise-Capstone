"""
Advanced ANC System - GUI Application
Cross-platform desktop application with AirPods-inspired features
Can be compiled to .exe using PyInstaller

Author: ANC Capstone Project
"""

import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QGroupBox, QTabWidget,
    QProgressBar, QTextEdit, QMessageBox, QStyle
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon
import logging
from datetime import datetime

# Import ANC system components
try:
    from advanced_anc_system import (
        MultiModeANCSystem, AudioMode, TransparencyConfig,
        HearingAidConfig, ConversationState
    )
    from audio_equalizer import AudioEqualizer, SpatialAudioSimulator
    ANC_AVAILABLE = True
except ImportError:
    ANC_AVAILABLE = False
    print("Warning: ANC modules not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessingThread(QThread):
    """Background thread for audio processing"""

    status_update = pyqtSignal(str)
    performance_update = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, anc_system):
        super().__init__()
        self.anc_system = anc_system
        self.is_running = False
        self.stream = None

    def run(self):
        """Run audio processing loop"""
        self.is_running = True

        try:
            def audio_callback(indata, outdata, frames, time_info, status):
                if status:
                    logger.warning(f"Stream status: {status}")

                try:
                    # Process audio
                    processed = self.anc_system.process_chunk(indata[:, 0])

                    # Ensure correct shape
                    if len(processed) < frames:
                        processed = np.pad(processed, (0, frames - len(processed)))
                    elif len(processed) > frames:
                        processed = processed[:frames]

                    # Output
                    outdata[:, 0] = processed

                    # Emit performance stats periodically
                    if hasattr(self, '_callback_count'):
                        self._callback_count += 1
                        if self._callback_count % 100 == 0:  # Every 100 callbacks
                            stats = self.anc_system.get_performance_stats()
                            self.performance_update.emit(stats)
                    else:
                        self._callback_count = 0

                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    outdata[:, 0] = indata[:, 0]  # Fallback to passthrough

            # Start audio stream
            with sd.Stream(
                samplerate=self.anc_system.sample_rate,
                blocksize=self.anc_system.block_size,
                channels=1,
                callback=audio_callback
            ):
                self.status_update.emit("Audio processing active")

                # Keep thread alive
                while self.is_running:
                    self.msleep(100)

        except Exception as e:
            self.error_occurred.emit(f"Audio processing error: {str(e)}")
            logger.error(f"Audio thread error: {e}")
        finally:
            self.status_update.emit("Audio processing stopped")

    def stop(self):
        """Stop audio processing"""
        self.is_running = False


class ANCControlPanel(QMainWindow):
    """Main GUI window for Advanced ANC System"""

    def __init__(self):
        super().__init__()

        self.anc_system = None
        self.processing_thread = None
        self.equalizer = None
        self.spatial_audio = None

        self.init_ui()
        self.init_anc_system()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Advanced ANC System - AirPods-Inspired Technology")
        self.setGeometry(100, 100, 900, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title_label = QLabel("🎧 Advanced Active Noise Cancellation System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        subtitle_label = QLabel("AirPods-Inspired Technology with AI Enhancement")
        subtitle_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle_label)

        # Tab widget for different sections
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Tab 1: Main Controls
        tabs.addTab(self.create_main_controls_tab(), "Main Controls")

        # Tab 2: Equalizer
        tabs.addTab(self.create_equalizer_tab(), "Equalizer")

        # Tab 3: Spatial Audio
        tabs.addTab(self.create_spatial_audio_tab(), "Spatial Audio")

        # Tab 4: Advanced Settings
        tabs.addTab(self.create_advanced_settings_tab(), "Advanced Settings")

        # Status bar
        self.status_label = QLabel("Status: Initializing...")
        main_layout.addWidget(self.status_label)

        # Performance bar
        self.performance_label = QLabel("Latency: -- ms | CPU: -- %")
        main_layout.addWidget(self.performance_label)

        # Log area
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

    def create_main_controls_tab(self) -> QWidget:
        """Create main controls tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Mode selection
        mode_group = QGroupBox("Audio Mode")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["ANC", "Transparency", "Adaptive", "Off"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("Select Mode:"))
        mode_layout.addWidget(self.mode_combo)

        mode_description = QLabel(
            "• ANC: Active noise cancellation\n"
            "• Transparency: Ambient sound passthrough\n"
            "• Adaptive: Auto-switch based on voice detection\n"
            "• Off: No processing (passthrough)"
        )
        mode_layout.addWidget(mode_description)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # ANC Intensity
        anc_group = QGroupBox("ANC Intensity")
        anc_layout = QVBoxLayout()

        self.anc_level_combo = QComboBox()
        self.anc_level_combo.addItems(["Gentle", "Normal", "Moderate", "Aggressive", "Maximum"])
        self.anc_level_combo.setCurrentText("Normal")
        self.anc_level_combo.currentTextChanged.connect(self.on_anc_level_changed)
        anc_layout.addWidget(QLabel("Noise Reduction Level:"))
        anc_layout.addWidget(self.anc_level_combo)

        anc_group.setLayout(anc_layout)
        layout.addWidget(anc_group)

        # Transparency settings
        trans_group = QGroupBox("Transparency Settings")
        trans_layout = QVBoxLayout()

        # Amplification
        trans_layout.addWidget(QLabel("Amplification:"))
        self.trans_amp_slider = QSlider(Qt.Horizontal)
        self.trans_amp_slider.setMinimum(50)
        self.trans_amp_slider.setMaximum(200)
        self.trans_amp_slider.setValue(100)
        self.trans_amp_slider.setTickPosition(QSlider.TicksBelow)
        self.trans_amp_slider.setTickInterval(25)
        self.trans_amp_slider.valueChanged.connect(self.on_transparency_changed)
        self.trans_amp_label = QLabel("1.0x")
        trans_amp_layout = QHBoxLayout()
        trans_amp_layout.addWidget(self.trans_amp_slider)
        trans_amp_layout.addWidget(self.trans_amp_label)
        trans_layout.addLayout(trans_amp_layout)

        # Conversation Boost
        trans_layout.addWidget(QLabel("Conversation Boost:"))
        self.trans_conv_slider = QSlider(Qt.Horizontal)
        self.trans_conv_slider.setMinimum(100)
        self.trans_conv_slider.setMaximum(200)
        self.trans_conv_slider.setValue(130)
        self.trans_conv_slider.setTickPosition(QSlider.TicksBelow)
        self.trans_conv_slider.setTickInterval(25)
        self.trans_conv_slider.valueChanged.connect(self.on_transparency_changed)
        self.trans_conv_label = QLabel("1.3x")
        trans_conv_layout = QHBoxLayout()
        trans_conv_layout.addWidget(self.trans_conv_slider)
        trans_conv_layout.addWidget(self.trans_conv_label)
        trans_layout.addLayout(trans_conv_layout)

        # Ambient Reduction
        trans_layout.addWidget(QLabel("Ambient Noise Reduction:"))
        self.trans_ambient_slider = QSlider(Qt.Horizontal)
        self.trans_ambient_slider.setMinimum(0)
        self.trans_ambient_slider.setMaximum(100)
        self.trans_ambient_slider.setValue(30)
        self.trans_ambient_slider.setTickPosition(QSlider.TicksBelow)
        self.trans_ambient_slider.setTickInterval(25)
        self.trans_ambient_slider.valueChanged.connect(self.on_transparency_changed)
        self.trans_ambient_label = QLabel("30%")
        trans_ambient_layout = QHBoxLayout()
        trans_ambient_layout.addWidget(self.trans_ambient_slider)
        trans_ambient_layout.addWidget(self.trans_ambient_label)
        trans_layout.addLayout(trans_ambient_layout)

        trans_group.setLayout(trans_layout)
        layout.addWidget(trans_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.calibrate_button = QPushButton("🎤 Calibrate Noise Profile")
        self.calibrate_button.clicked.connect(self.on_calibrate)
        button_layout.addWidget(self.calibrate_button)

        self.start_button = QPushButton("▶ Start Processing")
        self.start_button.clicked.connect(self.on_start_stop)
        button_layout.addWidget(self.start_button)

        layout.addLayout(button_layout)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_equalizer_tab(self) -> QWidget:
        """Create equalizer controls tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Preset selection
        preset_group = QGroupBox("EQ Presets")
        preset_layout = QHBoxLayout()

        preset_layout.addWidget(QLabel("Preset:"))
        self.eq_preset_combo = QComboBox()
        self.eq_preset_combo.addItems([
            "Flat", "Bass Boost", "Vocal Clarity", "Treble Boost",
            "Balanced", "Podcast", "Music", "Classical"
        ])
        self.eq_preset_combo.currentTextChanged.connect(self.on_eq_preset_changed)
        preset_layout.addWidget(self.eq_preset_combo)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # EQ Bands
        bands_group = QGroupBox("Equalizer Bands")
        bands_layout = QVBoxLayout()

        self.eq_sliders = {}
        band_names = [
            ("Sub Bass", "sub_bass", "40 Hz"),
            ("Bass", "bass", "150 Hz"),
            ("Low Mid", "low_mid", "400 Hz"),
            ("Mid", "mid", "1 kHz"),
            ("High Mid", "high_mid", "3 kHz"),
            ("Presence", "presence", "5 kHz"),
            ("Brilliance", "brilliance", "10 kHz")
        ]

        for display_name, band_name, freq in band_names:
            band_layout = QHBoxLayout()
            band_layout.addWidget(QLabel(f"{display_name} ({freq}):"))

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-120)  # -12.0 dB
            slider.setMaximum(120)   # +12.0 dB
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(30)
            slider.valueChanged.connect(lambda v, b=band_name: self.on_eq_band_changed(b, v))

            label = QLabel("0.0 dB")
            self.eq_sliders[band_name] = (slider, label)

            band_layout.addWidget(slider)
            band_layout.addWidget(label)

            bands_layout.addLayout(band_layout)

        bands_group.setLayout(bands_layout)
        layout.addWidget(bands_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_spatial_audio_tab(self) -> QWidget:
        """Create spatial audio controls tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel(
            "Spatial Audio simulates 3D sound positioning.\n"
            "Adjust the position of the virtual sound source."
        )
        layout.addWidget(info_label)

        # Azimuth (horizontal position)
        layout.addWidget(QLabel("Azimuth (Left ← → Right):"))
        self.spatial_azimuth_slider = QSlider(Qt.Horizontal)
        self.spatial_azimuth_slider.setMinimum(-180)
        self.spatial_azimuth_slider.setMaximum(180)
        self.spatial_azimuth_slider.setValue(0)
        self.spatial_azimuth_slider.setTickPosition(QSlider.TicksBelow)
        self.spatial_azimuth_slider.setTickInterval(45)
        self.spatial_azimuth_label = QLabel("0°")
        azimuth_layout = QHBoxLayout()
        azimuth_layout.addWidget(self.spatial_azimuth_slider)
        azimuth_layout.addWidget(self.spatial_azimuth_label)
        layout.addLayout(azimuth_layout)

        # Elevation (vertical position)
        layout.addWidget(QLabel("Elevation (Below ← → Above):"))
        self.spatial_elevation_slider = QSlider(Qt.Horizontal)
        self.spatial_elevation_slider.setMinimum(-90)
        self.spatial_elevation_slider.setMaximum(90)
        self.spatial_elevation_slider.setValue(0)
        self.spatial_elevation_slider.setTickPosition(QSlider.TicksBelow)
        self.spatial_elevation_slider.setTickInterval(30)
        self.spatial_elevation_label = QLabel("0°")
        elevation_layout = QHBoxLayout()
        elevation_layout.addWidget(self.spatial_elevation_slider)
        elevation_layout.addWidget(self.spatial_elevation_label)
        layout.addLayout(elevation_layout)

        # Distance
        layout.addWidget(QLabel("Distance (Close ← → Far):"))
        self.spatial_distance_slider = QSlider(Qt.Horizontal)
        self.spatial_distance_slider.setMinimum(50)
        self.spatial_distance_slider.setMaximum(300)
        self.spatial_distance_slider.setValue(100)
        self.spatial_distance_slider.setTickPosition(QSlider.TicksBelow)
        self.spatial_distance_slider.setTickInterval(50)
        self.spatial_distance_label = QLabel("1.0")
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.spatial_distance_slider)
        distance_layout.addWidget(self.spatial_distance_label)
        layout.addLayout(distance_layout)

        # Update labels when sliders change
        self.spatial_azimuth_slider.valueChanged.connect(
            lambda v: self.spatial_azimuth_label.setText(f"{v}°")
        )
        self.spatial_elevation_slider.valueChanged.connect(
            lambda v: self.spatial_elevation_label.setText(f"{v}°")
        )
        self.spatial_distance_slider.valueChanged.connect(
            lambda v: self.spatial_distance_label.setText(f"{v/100:.1f}")
        )

        # Quick position presets
        preset_group = QGroupBox("Quick Positions")
        preset_layout = QVBoxLayout()

        positions = [
            ("Center", 0, 0, 100),
            ("Left", -90, 0, 100),
            ("Right", 90, 0, 100),
            ("Front Left", -45, 0, 100),
            ("Front Right", 45, 0, 100),
        ]

        for name, azimuth, elevation, distance in positions:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, a=azimuth, e=elevation, d=distance:
                self.set_spatial_position(a, e, d)
            )
            preset_layout.addWidget(btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_advanced_settings_tab(self) -> QWidget:
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # System info
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout()

        self.system_info_label = QLabel("Loading system info...")
        info_layout.addWidget(self.system_info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Performance stats
        perf_group = QGroupBox("Performance Statistics")
        perf_layout = QVBoxLayout()

        self.perf_stats_label = QLabel("No statistics available yet")
        perf_layout.addWidget(self.perf_stats_label)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # About
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout()

        about_text = QLabel(
            "<b>Advanced ANC System</b><br>"
            "Version 1.0<br><br>"
            "AirPods-inspired noise cancellation technology<br>"
            "featuring adaptive transparency, conversation awareness,<br>"
            "hearing aid functionality, and spatial audio simulation.<br><br>"
            "<b>Features:</b><br>"
            "• Multi-mode ANC (ANC/Transparency/Adaptive/Off)<br>"
            "• Voice activity detection<br>"
            "• 7-band parametric equalizer<br>"
            "• Spatial audio simulation<br>"
            "• Real-time processing (&lt;50ms latency)<br><br>"
            "© 2025 ANC Capstone Project"
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)

        about_group.setLayout(about_layout)
        layout.addWidget(about_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def init_anc_system(self):
        """Initialize ANC system"""
        if not ANC_AVAILABLE:
            self.log("Error: ANC system modules not available")
            QMessageBox.critical(
                self,
                "Error",
                "ANC system modules could not be loaded.\n"
                "Please ensure all dependencies are installed."
            )
            return

        try:
            self.anc_system = MultiModeANCSystem(
                sample_rate=44100,
                block_size=2048,
                mode=AudioMode.ANC,
                noise_reduction_level="normal"
            )

            self.equalizer = AudioEqualizer(sample_rate=44100)
            self.spatial_audio = SpatialAudioSimulator(sample_rate=44100)

            self.log("✓ ANC system initialized successfully")
            self.update_system_info()

        except Exception as e:
            self.log(f"Error initializing ANC system: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize ANC system:\n{str(e)}")

    def update_system_info(self):
        """Update system information display"""
        if self.anc_system:
            info = (
                f"Sample Rate: {self.anc_system.sample_rate} Hz\n"
                f"Block Size: {self.anc_system.block_size} samples\n"
                f"Latency: ~{self.anc_system.block_size / self.anc_system.sample_rate * 1000:.1f} ms\n"
                f"Mode: {self.anc_system.mode.value}\n"
                f"Calibrated: {self.anc_system.is_calibrated}"
            )
            self.system_info_label.setText(info)

    def on_mode_changed(self, mode_text: str):
        """Handle mode change"""
        if not self.anc_system:
            return

        mode_map = {
            "ANC": AudioMode.ANC,
            "Transparency": AudioMode.TRANSPARENCY,
            "Adaptive": AudioMode.ADAPTIVE,
            "Off": AudioMode.OFF
        }

        mode = mode_map.get(mode_text)
        if mode:
            self.anc_system.set_mode(mode)
            self.log(f"Mode changed to: {mode_text}")
            self.update_system_info()

    def on_anc_level_changed(self, level_text: str):
        """Handle ANC level change"""
        if not self.anc_system:
            return

        self.anc_system.noise_reduction_level = level_text.lower()
        self.log(f"ANC level changed to: {level_text}")

    def on_transparency_changed(self):
        """Handle transparency settings change"""
        if not self.anc_system:
            return

        amp = self.trans_amp_slider.value() / 100.0
        conv = self.trans_conv_slider.value() / 100.0
        ambient = self.trans_ambient_slider.value() / 100.0

        self.trans_amp_label.setText(f"{amp:.1f}x")
        self.trans_conv_label.setText(f"{conv:.1f}x")
        self.trans_ambient_label.setText(f"{ambient * 100:.0f}%")

        # Update transparency config
        self.anc_system.transparency.config.amplification = amp
        self.anc_system.transparency.config.conversation_boost = conv
        self.anc_system.transparency.config.ambient_reduction = ambient

    def on_eq_preset_changed(self, preset_text: str):
        """Handle EQ preset change"""
        if not self.equalizer:
            return

        preset_name = preset_text.lower().replace(" ", "_")
        self.equalizer.apply_preset(preset_name)
        self.log(f"EQ preset applied: {preset_text}")

        # Update sliders to match preset
        for band_name, (slider, label) in self.eq_sliders.items():
            if band_name in self.equalizer.bands:
                gain = self.equalizer.bands[band_name].gain
                slider.setValue(int(gain * 10))
                label.setText(f"{gain:.1f} dB")

    def on_eq_band_changed(self, band_name: str, value: int):
        """Handle individual EQ band change"""
        if not self.equalizer:
            return

        gain_db = value / 10.0
        self.equalizer.set_band_gain(band_name, gain_db)

        # Update label
        if band_name in self.eq_sliders:
            _, label = self.eq_sliders[band_name]
            label.setText(f"{gain_db:.1f} dB")

    def set_spatial_position(self, azimuth: int, elevation: int, distance: int):
        """Set spatial audio position"""
        self.spatial_azimuth_slider.setValue(azimuth)
        self.spatial_elevation_slider.setValue(elevation)
        self.spatial_distance_slider.setValue(distance)

    def on_calibrate(self):
        """Handle calibration request"""
        if not self.anc_system:
            return

        reply = QMessageBox.question(
            self,
            "Calibrate Noise Profile",
            "Calibration will record background noise for 3 seconds.\n"
            "Please remain silent and minimize background noise.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self.log("Starting noise calibration...")
                self.calibrate_button.setEnabled(False)
                self.status_label.setText("Status: Calibrating...")

                QApplication.processEvents()

                self.anc_system.calibrate_noise(duration=3.0)

                self.log("✓ Calibration complete")
                self.status_label.setText("Status: Calibrated")
                self.update_system_info()

            except Exception as e:
                self.log(f"Calibration error: {e}")
                QMessageBox.warning(self, "Error", f"Calibration failed:\n{str(e)}")
            finally:
                self.calibrate_button.setEnabled(True)

    def on_start_stop(self):
        """Handle start/stop button"""
        if not self.anc_system:
            return

        if self.processing_thread and self.processing_thread.isRunning():
            # Stop processing
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.start_button.setText("▶ Start Processing")
            self.log("Processing stopped")
        else:
            # Check if calibrated for ANC mode
            if self.anc_system.mode == AudioMode.ANC and not self.anc_system.is_calibrated:
                QMessageBox.warning(
                    self,
                    "Not Calibrated",
                    "Please calibrate the noise profile before using ANC mode."
                )
                return

            # Start processing
            self.processing_thread = AudioProcessingThread(self.anc_system)
            self.processing_thread.status_update.connect(self.on_status_update)
            self.processing_thread.performance_update.connect(self.on_performance_update)
            self.processing_thread.error_occurred.connect(self.on_error)
            self.processing_thread.start()
            self.start_button.setText("■ Stop Processing")
            self.log("Processing started")

    def on_status_update(self, status: str):
        """Handle status update from processing thread"""
        self.status_label.setText(f"Status: {status}")

    def on_performance_update(self, stats: dict):
        """Handle performance update from processing thread"""
        latency = stats.get('avg_latency_ms', 0)
        self.performance_label.setText(f"Latency: {latency:.2f} ms")

        # Update performance stats in advanced tab
        stats_text = ""
        for key, value in stats.items():
            stats_text += f"{key}: {value}\n"
        self.perf_stats_label.setText(stats_text)

    def on_error(self, error_msg: str):
        """Handle error from processing thread"""
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)

    def closeEvent(self, event):
        """Handle window close"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = ANCControlPanel()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
