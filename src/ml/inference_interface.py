"""
Inference Interface for Consciousness Models

This module provides testing interfaces for Mode 1 (RNG only) and 
Mode 2 (RNG+EEG) output generation using trained models.
"""

import sys
import os
# Add the project root to sys.path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import threading
import queue
import json
import math
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QComboBox,
                                QGroupBox, QSlider, QTextEdit, QSplitter, QTabWidget,
                                QProgressBar, QCheckBox)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# Handle imports with proper fallback for different entry points
try:
    from .training_pipeline import TrainingConfig
except ImportError:
    from src.ml.training_pipeline import TrainingConfig

try:
    # Prefer absolute import; project adds 'src' to sys.path
    from utils.curve_3d import InterlockingDialSystem
except ImportError:
    from src.utils.curve_3d import InterlockingDialSystem

try:
    from gui.painting_interface import PaintCanvas, DataVisualizationWidget
except ImportError:
    from src.gui.painting_interface import PaintCanvas, DataVisualizationWidget

try:
    from gui.field_visualization import create_mystical_field_widget, MysticalFieldWidget
except ImportError:
    from src.gui.field_visualization import create_mystical_field_widget, MysticalFieldWidget

try:
    from .inference_network import InferenceStreamServer, StreamConfig
except ImportError:
    from src.ml.inference_network import InferenceStreamServer, StreamConfig

try:
    from .model_manager import ModelManager
    from .advanced_inference import AdvancedInferenceEngine, MultiModelInferenceConfig, PredictionResult
except ImportError:
    from src.ml.model_manager import ModelManager
    from src.ml.advanced_inference import AdvancedInferenceEngine, MultiModelInferenceConfig, PredictionResult


@dataclass
class InferenceConfig:
    """Configuration for inference mode"""
    mode: int = 1  # 1 for RNG only, 2 for RNG+EEG
    model_path: str = ""
    prediction_rate: float = 10.0  # Hz
    sequence_length: int = 100
    real_time: bool = True
    auto_generate: bool = False
    generation_interval: float = 2.0  # seconds
    # Network streaming options
    enable_streaming: bool = False
    stream_port: int = 8765
    stream_id: str = "inference_1"


@dataclass
class InferenceEngine:
    """Core inference engine for consciousness models"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        # Import ModelTrainer locally to avoid circular imports
        try:
            from .training_pipeline import ModelTrainer
        except ImportError:
            from ml.training_pipeline import ModelTrainer
        self.model_trainer = ModelTrainer()
        self.dial_system = InterlockingDialSystem()
        
        # Data buffers for sequence prediction
        self.rng_buffer = queue.Queue(maxsize=config.sequence_length * 2)
        self.eeg_buffer = queue.Queue(maxsize=config.sequence_length * 2)
        
        # Prediction state
        self.is_running = False
        self.prediction_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks for real-time updates
        self.prediction_callbacks: List[Callable[[PredictionResult], None]] = []
        
        # Network streaming
        self.stream_server: Optional[InferenceStreamServer] = None
        if config.enable_streaming:
            stream_config = StreamConfig(
                port=config.stream_port,
                stream_id=config.stream_id,
                mode=config.mode
            )
            self.stream_server = InferenceStreamServer(stream_config)
        
        # Statistics
        self.total_predictions = 0
        self.prediction_times = []
        
    def load_model(self, model_path: str):
        """Load trained model for inference"""
        try:
            if self.config.mode == 1:
                self.model_trainer.load_models(mode1_path=model_path)
            else:
                self.model_trainer.load_models(mode2_path=model_path)
            print(f"Loaded model for mode {self.config.mode}: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def add_rng_data(self, rng_values: List[float]):
        """Add RNG data to the buffer"""
        try:
            self.rng_buffer.put_nowait(rng_values)
        except queue.Full:
            # Remove oldest entry
            try:
                self.rng_buffer.get_nowait()
                self.rng_buffer.put_nowait(rng_values)
            except queue.Empty:
                pass
                
    def add_eeg_data(self, eeg_channels: Dict[str, float]):
        """Add EEG data to the buffer"""
        if self.config.mode == 2:
            try:
                self.eeg_buffer.put_nowait(eeg_channels)
            except queue.Full:
                # Remove oldest entry
                try:
                    self.eeg_buffer.get_nowait()
                    self.eeg_buffer.put_nowait(eeg_channels)
                except queue.Empty:
                    pass
                    
    def start_inference(self):
        """Start real-time inference"""
        if self.is_running:
            return
            
        self.stop_event.clear()
        self.is_running = True
        
        # Start network streaming if enabled
        if self.stream_server:
            try:
                self.stream_server.start()
                print(f"Started inference stream on port {self.stream_server.config.port}")
            except Exception as e:
                print(f"Warning: Failed to start inference stream: {e}")
        
        self.prediction_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True
        )
        self.prediction_thread.start()
        
        print(f"Started inference mode {self.config.mode}")
        
    def stop_inference(self):
        """Stop real-time inference"""
        if not self.is_running:
            return
            
        self.stop_event.set()
        self.is_running = False
        
        if self.prediction_thread:
            self.prediction_thread.join(timeout=2.0)
            
        # Stop network streaming
        if self.stream_server:
            try:
                self.stream_server.stop()
                print("Stopped inference stream")
            except Exception as e:
                print(f"Warning: Error stopping inference stream: {e}")
            
        print("Stopped inference")
        
    def _inference_worker(self):
        """Background worker for continuous inference"""
        prediction_interval = 1.0 / self.config.prediction_rate
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                prediction = self._make_prediction()
                if prediction:
                    # Call callbacks
                    for callback in self.prediction_callbacks:
                        try:
                            callback(prediction)
                        except Exception as e:
                            print(f"Error in prediction callback: {e}")
                    
                    # Send to network stream if enabled
                    if self.stream_server:
                        try:
                            self.stream_server.send_prediction(prediction)
                        except Exception as e:
                            print(f"Error streaming prediction: {e}")
                            
                    self.total_predictions += 1
                    
            except Exception as e:
                print(f"Error in inference worker: {e}")
                
            # Maintain prediction rate
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)
            
            sleep_time = max(0, prediction_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _make_prediction(self) -> Optional[PredictionResult]:
        """Make a single prediction"""
        
        # Check if we have enough data
        rng_data = self._get_rng_sequence()
        if rng_data is None:
            return None
            
        eeg_data = None
        if self.config.mode == 2:
            eeg_data = self._get_eeg_sequence()
            if eeg_data is None:
                return None
                
        try:
            # Make prediction using the model
            predictions = self.model_trainer.predict(
                rng_data=rng_data,
                eeg_data=eeg_data,
                mode=self.config.mode
            )
            
            # Create result
            result = PredictionResult(
                timestamp=time.time(),
                mode=self.config.mode,
                colors=predictions['colors'],
                curves=predictions['curves'],
                dials=predictions['dials'],
                confidence=self._calculate_confidence(predictions),
                input_data={
                    'rng': rng_data,
                    'eeg': eeg_data
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
            
    def _get_rng_sequence(self) -> Optional[np.ndarray]:
        """Get RNG sequence from buffer"""
        if self.rng_buffer.qsize() < self.config.sequence_length:
            return None
            
        # Extract sequence
        sequence_data = []
        temp_buffer = []
        
        # Get all data from buffer
        while not self.rng_buffer.empty():
            try:
                data = self.rng_buffer.get_nowait()
                temp_buffer.append(data)
            except queue.Empty:
                break
                
        # Take last sequence_length items
        if len(temp_buffer) >= self.config.sequence_length:
            sequence_data = temp_buffer[-self.config.sequence_length:]
            
            # Put back extra data
            for item in temp_buffer[:-self.config.sequence_length]:
                try:
                    self.rng_buffer.put_nowait(item)
                except queue.Full:
                    break
                    
        else:
            # Put all data back
            for item in temp_buffer:
                try:
                    self.rng_buffer.put_nowait(item)
                except queue.Full:
                    break
            return None
            
        # Convert to numpy array
        return np.array(sequence_data)
        
    def _get_eeg_sequence(self) -> Optional[np.ndarray]:
        """Get EEG sequence from buffer"""
        if self.eeg_buffer.qsize() < self.config.sequence_length:
            return None
            
        # Extract sequence
        sequence_data = []
        temp_buffer = []
        
        # Get all data from buffer
        while not self.eeg_buffer.empty():
            try:
                data = self.eeg_buffer.get_nowait()
                temp_buffer.append(data)
            except queue.Empty:
                break
                
        # Take last sequence_length items
        if len(temp_buffer) >= self.config.sequence_length:
            sequence_data = temp_buffer[-self.config.sequence_length:]
            
            # Put back extra data
            for item in temp_buffer[:-self.config.sequence_length]:
                try:
                    self.eeg_buffer.put_nowait(item)
                except queue.Full:
                    break
        else:
            # Put all data back
            for item in temp_buffer:
                try:
                    self.eeg_buffer.put_nowait(item)
                except queue.Full:
                    break
            return None
            
        # Convert to numpy array - extract channel values in consistent order
        eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        channel_sequences = []
        for eeg_data in sequence_data:
            channel_values = []
            for channel in eeg_channels:
                channel_values.append(eeg_data.get(channel, 0.0))
            channel_sequences.append(channel_values)
            
        return np.array(channel_sequences)
        
    def _calculate_confidence(self, predictions: Dict[str, np.ndarray]) -> float:
        """Calculate confidence score for predictions"""
        # Simple confidence metric based on prediction variance
        # In practice, you might use model uncertainty estimation
        
        color_var = np.var(predictions['colors'])
        curve_var = np.var(predictions['curves'])
        dial_var = np.var(predictions['dials'])
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + color_var + curve_var + dial_var)
        return min(1.0, max(0.0, confidence))
        
    def add_prediction_callback(self, callback: Callable[[PredictionResult], None]):
        """Add callback for prediction results"""
        self.prediction_callbacks.append(callback)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = np.mean(self.prediction_times[-100:]) if self.prediction_times else 0
        
        return {
            'total_predictions': self.total_predictions,
            'average_prediction_time': avg_time,
            'current_rate': 1.0 / avg_time if avg_time > 0 else 0,
            'buffer_sizes': {
                'rng': self.rng_buffer.qsize(),
                'eeg': self.eeg_buffer.qsize()
            }
        }


class InferenceWindow(QMainWindow):
    """Main window for inference/testing interface"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Consciousness Model Inference")
        self.setGeometry(100, 100, 1400, 900)
        
        # Legacy inference engines (kept for compatibility)
        self.inference_engines = {}
        self.current_engine: Optional[InferenceEngine] = None
        
        # New multi-model system
        self.current_advanced_engine = None
        self.current_model_name = None
        
        # Hardware devices (injected from main app)
        self.rng_device = None
        self.eeg_device = None
        
        # Streaming configuration
        self.stream_port = 8765
        self.stream_id = "inference_1"
        
        # Generated outputs
        self.generated_canvas = None
        self.dial_system = InterlockingDialSystem()
        
        # Recursive inference system
        self.recursive_mode = False
        self.recursive_depth = 2
        self.recursive_layers = []  # Stack of inference layers
        self.feedback_buffer = queue.Queue(maxsize=1000)
        
        self.setup_ui()
        self.setup_timers()
        
        # Initialize demo mode
        self.initialize_demo_mode()
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with horizontal split
        layout = QVBoxLayout(central_widget)
        
        # Control panel (at top)
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Main content area with horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Tabs for inference, generation, stats
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        
        # Inference tab
        inference_tab = self.create_inference_tab()
        self.tab_widget.addTab(inference_tab, "Real-time Inference")
        
        # Generation tab
        generation_tab = self.create_generation_tab()
        self.tab_widget.addTab(generation_tab, "Generated Output")
        
        # Statistics tab
        stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistics")
        
        left_layout.addWidget(self.tab_widget)
        main_splitter.addWidget(left_widget)
        
        # Right side: Dedicated mystical field area
        right_widget = self.create_dedicated_field_area()
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions (70% left, 30% right)
        main_splitter.setSizes([700, 300])
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(main_splitter)
        
    def create_control_panel(self) -> QWidget:
        """Create the main control panel"""
        panel = QWidget()
        panel.setMinimumHeight(200)  # Increased to accommodate content
        panel.setMaximumHeight(250)  # Increased maximum height
        layout = QHBoxLayout(panel)
        
        # Create status label first (before populate_model_list is called)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                color: #2e7d32;
                padding: 8px;
                border: 2px solid #c8e6c9;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        
        # Mode selection
        mode_group = QGroupBox("Model Selection")
        mode_group.setMinimumHeight(180)  # Reduced from 350 to fit in panel
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(8)  # Reduced spacing to fit better
        
        # Model selector section
        model_label = QLabel("Available Models:")
        model_label.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout.addWidget(model_label)
        
        # Model selector dropdown - make it larger
        self.model_selector = QComboBox()
        self.model_selector.setMinimumHeight(35)  # Larger dropdown
        self.model_selector.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 12px;
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                width: 30px;
            }
        """)
        self.model_selector.addItem("Select a trained model...")
        self.populate_model_list()
        self.model_selector.currentIndexChanged.connect(self.model_changed)
        mode_layout.addWidget(self.model_selector)
        
        # Refresh models button - make it more prominent
        self.refresh_models_button = QPushButton("🔄 Refresh Models")
        self.refresh_models_button.setMinimumHeight(35)
        self.refresh_models_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #4CAF50;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.refresh_models_button.clicked.connect(self.populate_model_list)
        mode_layout.addWidget(self.refresh_models_button)
        
        # Add some spacing
        mode_layout.addSpacing(10)
        
        # Model info section
        info_label = QLabel("Model Information:")
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        mode_layout.addWidget(info_label)
        
        # Model info display - make it more spacious
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setMinimumHeight(120)  # Ensure adequate height for info
        self.model_info_label.setStyleSheet("""
            QLabel { 
                background-color: #f8f9fa; 
                padding: 12px; 
                border: 2px solid #e9ecef; 
                border-radius: 8px;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        mode_layout.addWidget(self.model_info_label)
        
        # Inference controls
        control_group = QGroupBox("Inference Control")
        control_group.setMinimumHeight(150)
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(12)
        
        # Start button - make it prominent
        self.start_button = QPushButton("▶️ Start Inference")
        self.start_button.setMinimumHeight(45)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_button.clicked.connect(self.start_inference)
        self.start_button.setEnabled(False)  # Disabled until model is selected
        control_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("⏹️ Stop Inference")
        self.stop_button.setMinimumHeight(45)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Prediction rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Rate (Hz):"))
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(1, 50)
        self.rate_slider.setValue(10)
        self.rate_label = QLabel("10")
        self.rate_slider.valueChanged.connect(lambda v: self.rate_label.setText(str(v)))
        rate_layout.addWidget(self.rate_slider)
        rate_layout.addWidget(self.rate_label)
        
        # Auto-generation
        self.auto_generate_check = QCheckBox("Auto-generate")
        self.auto_generate_check.setChecked(True)  # Enable by default
        
        # Network streaming
        self.streaming_check = QCheckBox("Enable Streaming")
        
        # Recursive Inference Mode
        self.recursive_check = QCheckBox("Recursive Inference")
        self.recursive_check.setToolTip("Run inference under generation to create consciousness feedback loops")
        self.recursive_check.stateChanged.connect(self.toggle_recursive_mode)
        
        # Recursive depth control
        recursive_depth_layout = QHBoxLayout()
        recursive_depth_layout.addWidget(QLabel("Recursion Depth:"))
        self.recursive_depth_slider = QSlider(Qt.Horizontal)
        self.recursive_depth_slider.setRange(1, 5)
        self.recursive_depth_slider.setValue(2)
        self.recursive_depth_label = QLabel("2")
        self.recursive_depth_slider.valueChanged.connect(lambda v: self.recursive_depth_label.setText(str(v)))
        self.recursive_depth_slider.valueChanged.connect(lambda v: setattr(self, 'recursive_depth', v))
        self.recursive_depth_slider.valueChanged.connect(self.rebuild_recursive_layers)
        self.recursive_depth_slider.setEnabled(False)  # Disabled until recursive mode is enabled
        recursive_depth_layout.addWidget(self.recursive_depth_slider)
        recursive_depth_layout.addWidget(self.recursive_depth_label)
        
        settings_layout.addLayout(rate_layout)
        settings_layout.addWidget(self.auto_generate_check)
        settings_layout.addWidget(self.streaming_check)
        settings_layout.addWidget(self.recursive_check)
        settings_layout.addLayout(recursive_depth_layout)
        
        # Status
        status_group = QGroupBox("Status")
        status_group.setMinimumHeight(100)
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        
        # Add the status label that was created earlier
        status_layout.addWidget(self.status_label)
        
        # Prediction count with styling
        self.prediction_count_label = QLabel("Predictions: 0")
        self.prediction_count_label.setStyleSheet("""
            QLabel {
                background-color: #f3f4f6;
                padding: 6px;
                border: 1px solid #d1d5db;
                border-radius: 3px;
                font-size: 11px;
            }
        """)
        status_layout.addWidget(self.prediction_count_label)
        
        # Add all groups
        layout.addWidget(mode_group)
        layout.addWidget(control_group)
        layout.addWidget(settings_group)
        layout.addWidget(status_group)
        layout.addStretch()
        
        return panel
        
    def create_inference_tab(self) -> QWidget:
        """Create the real-time inference tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel - Input data visualization
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)  # Set minimum width
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Input Data Streams"))
        self.input_viz = DataVisualizationWidget(400, 300)
        left_layout.addWidget(self.input_viz)
        
        left_layout.addWidget(QLabel("Prediction Confidence"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setMinimumHeight(25)  # Make progress bar taller
        left_layout.addWidget(self.confidence_bar)
        
        # Right panel - Prediction outputs
        right_panel = QWidget()
        right_panel.setMinimumWidth(350)  # Set minimum width
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("Generated Colors"))
        self.color_display = QWidget()
        self.color_display.setMinimumHeight(120)  # Increased from 50
        self.color_display.setMaximumHeight(150)  # Set reasonable max
        self.color_display.setStyleSheet("background-color: black; border: 2px solid #444; border-radius: 5px;")
        right_layout.addWidget(self.color_display)
        
        right_layout.addWidget(QLabel("3D Dial System"))
        # Placeholder for 3D visualization (would need OpenGL widget)
        self.dial_display = QLabel("3D Dial Visualization\n(Requires OpenGL)")
        self.dial_display.setStyleSheet("border: 2px solid gray; background-color: #222; border-radius: 5px; color: #ccc; font-size: 12px;")
        self.dial_display.setMinimumHeight(250)  # Reduced from 300
        self.dial_display.setMinimumWidth(300)   # Added minimum width
        self.dial_display.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.dial_display)
        
        layout.addWidget(left_panel, 1)  # Give left panel stretch factor 1
        layout.addWidget(right_panel, 1) # Give right panel stretch factor 1 (equal sizing)
        
        return tab
    
    def create_field_tab(self) -> QWidget:
        """Create the mystical field visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description label
        desc_label = QLabel("🔮 Mystical Field - Physical particle simulation driven by consciousness RNG stream")
        desc_label.setStyleSheet("""
            QLabel {
                color: #9932CC;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(153, 50, 204, 0.1);
                border: 1px solid #9932CC;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(desc_label)
        
        # Create mystical field widget
        try:
            self.field_widget = create_mystical_field_widget()
            if self.field_widget:
                layout.addWidget(self.field_widget)
                
                # Field controls
                controls_layout = QHBoxLayout()
                
                # Field intensity control
                intensity_label = QLabel("Field Intensity:")
                intensity_label.setStyleSheet("color: #DDA0DD; font-weight: bold;")
                controls_layout.addWidget(intensity_label)
                
                self.field_intensity_slider = QSlider(Qt.Horizontal)
                self.field_intensity_slider.setRange(10, 200)
                self.field_intensity_slider.setValue(100)
                self.field_intensity_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #9932CC;
                        height: 8px;
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                  stop:0 #2F1B69, stop:1 #9932CC);
                        border-radius: 4px;
                    }
                    QSlider::handle:horizontal {
                        background: #FF1493;
                        border: 2px solid #9932CC;
                        width: 18px;
                        border-radius: 9px;
                    }
                """)
                controls_layout.addWidget(self.field_intensity_slider)
                
                # Particle count display
                self.particle_count_label = QLabel("Particles: 0")
                self.particle_count_label.setStyleSheet("color: #00FFFF; font-weight: bold;")
                controls_layout.addWidget(self.particle_count_label)
                
                controls_layout.addStretch()
                
                # Reset field button
                reset_field_button = QPushButton("🌟 Reset Field")
                reset_field_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4B0082;
                        color: #FFD700;
                        border: 2px solid #9932CC;
                        border-radius: 8px;
                        padding: 8px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #6A0DAD;
                    }
                    QPushButton:pressed {
                        background-color: #3A006A;
                    }
                """)
                reset_field_button.clicked.connect(self.reset_mystical_field)
                controls_layout.addWidget(reset_field_button)
                
                layout.addLayout(controls_layout)
                
                # Connect intensity slider
                self.field_intensity_slider.valueChanged.connect(self.update_field_intensity)
                
            else:
                # Fallback if PyQt5 not available
                fallback_label = QLabel("Mystical Field requires PyQt5 for visualization")
                fallback_label.setAlignment(Qt.AlignCenter)
                fallback_label.setStyleSheet("color: #FF6347; font-size: 16px;")
                layout.addWidget(fallback_label)
                self.field_widget = None
                
        except Exception as e:
            error_label = QLabel(f"Error creating mystical field: {e}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #FF6347; font-size: 14px;")
            layout.addWidget(error_label)
            self.field_widget = None
        
        return tab
    
    def create_dedicated_field_area(self) -> QWidget:
        """Create the dedicated mystical field side area"""
        field_widget = QWidget()
        field_widget.setMinimumWidth(350)
        layout = QVBoxLayout(field_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title with mystical styling
        title_label = QLabel("🔮 Mystical Consciousness Field")
        title_label.setStyleSheet("""
            QLabel {
                color: #9932CC;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 rgba(153, 50, 204, 0.2), 
                                          stop:1 rgba(75, 0, 130, 0.2));
                border: 2px solid #9932CC;
                border-radius: 8px;
                margin-bottom: 8px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create mystical field widget
        try:
            self.field_widget = create_mystical_field_widget()
            if self.field_widget:
                # Set fixed size for the field
                self.field_widget.setFixedSize(330, 400)
                layout.addWidget(self.field_widget)
                
                # Compact field controls
                controls_group = QGroupBox("Field Controls")
                controls_group.setStyleSheet("""
                    QGroupBox {
                        color: #DDA0DD;
                        font-weight: bold;
                        border: 1px solid #9932CC;
                        border-radius: 5px;
                        margin-top: 6px;
                        padding-top: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px 0 5px;
                    }
                """)
                controls_layout = QVBoxLayout(controls_group)
                controls_layout.setSpacing(6)
                
                # Intensity control row
                intensity_row = QHBoxLayout()
                intensity_label = QLabel("Intensity:")
                intensity_label.setStyleSheet("color: #DDA0DD; font-weight: bold; font-size: 11px;")
                intensity_label.setFixedWidth(60)
                intensity_row.addWidget(intensity_label)
                
                self.field_intensity_slider = QSlider(Qt.Horizontal)
                self.field_intensity_slider.setRange(10, 200)
                self.field_intensity_slider.setValue(100)
                self.field_intensity_slider.setFixedHeight(20)
                self.field_intensity_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #9932CC;
                        height: 6px;
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                  stop:0 #2F1B69, stop:1 #9932CC);
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #FF1493;
                        border: 1px solid #9932CC;
                        width: 14px;
                        border-radius: 7px;
                    }
                """)
                intensity_row.addWidget(self.field_intensity_slider)
                controls_layout.addLayout(intensity_row)
                
                # Particle count and reset row
                stats_row = QHBoxLayout()
                self.particle_count_label = QLabel("Particles: 0")
                self.particle_count_label.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 11px;")
                stats_row.addWidget(self.particle_count_label)
                
                stats_row.addStretch()
                
                reset_field_button = QPushButton("🌟 Reset")
                reset_field_button.setFixedSize(70, 25)
                reset_field_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4B0082;
                        color: #FFD700;
                        border: 1px solid #9932CC;
                        border-radius: 4px;
                        font-size: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #6A0DAD;
                    }
                    QPushButton:pressed {
                        background-color: #3A006A;
                    }
                """)
                reset_field_button.clicked.connect(self.reset_mystical_field)
                stats_row.addWidget(reset_field_button)
                
                controls_layout.addLayout(stats_row)
                layout.addWidget(controls_group)
                
                # Connect intensity slider
                self.field_intensity_slider.valueChanged.connect(self.update_field_intensity)
                
            else:
                # Fallback if PyQt5 not available
                fallback_label = QLabel("🔮 Field Unavailable\n\nPyQt5 required for\nmystical visualization")
                fallback_label.setAlignment(Qt.AlignCenter)
                fallback_label.setStyleSheet("""
                    QLabel {
                        color: #FF6347; 
                        font-size: 14px;
                        background-color: rgba(255, 99, 71, 0.1);
                        border: 1px solid #FF6347;
                        border-radius: 8px;
                        padding: 20px;
                    }
                """)
                layout.addWidget(fallback_label)
                self.field_widget = None
                
        except Exception as e:
            error_label = QLabel(f"🔮 Field Error\n\n{str(e)[:50]}...")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("""
                QLabel {
                    color: #FF6347; 
                    font-size: 12px;
                    background-color: rgba(255, 99, 71, 0.1);
                    border: 1px solid #FF6347;
                    border-radius: 8px;
                    padding: 20px;
                }
            """)
            layout.addWidget(error_label)
            self.field_widget = None
        
        layout.addStretch()
        return field_widget
        
    def create_generation_tab(self) -> QWidget:
        """Create the generation output tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Canvas for generated artwork
        layout.addWidget(QLabel("Generated Artwork"))
        self.generated_canvas = PaintCanvas(800, 600)
        layout.addWidget(self.generated_canvas)
        
        # Generation controls
        gen_controls = QHBoxLayout()
        
        self.generate_button = QPushButton("Generate Single Output")
        self.generate_button.clicked.connect(self.generate_single)
        
        # Add a test button for immediate verification
        self.test_generate_button = QPushButton("Test Generate (Debug)")
        self.test_generate_button.clicked.connect(self.test_generate_immediate)
        
        self.clear_generated_button = QPushButton("Clear Generated")
        self.clear_generated_button.clicked.connect(self.clear_generated)
        
        self.save_generated_button = QPushButton("Save Generated")
        self.save_generated_button.clicked.connect(self.save_generated)
        
        gen_controls.addWidget(self.generate_button)
        gen_controls.addWidget(self.test_generate_button)
        gen_controls.addWidget(self.clear_generated_button)
        gen_controls.addWidget(self.save_generated_button)
        gen_controls.addStretch()
        
        layout.addLayout(gen_controls)
        
        return tab
        
    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.stats_text)
        
        return tab
        
    def setup_timers(self):
        """Set up update timers"""
        # UI update timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10 Hz UI updates
        
        # Statistics update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(1000)  # 1 Hz stats updates
        
        # Mystical field simulation timer (for when no real RNG data)
        self.field_sim_timer = QTimer()
        self.field_sim_timer.timeout.connect(self.simulate_field_data)
        self.field_sim_timer.start(50)  # 20 Hz field simulation
        
        # Demo inference simulation timer (for testing the Real-time Inference tab)
        self.demo_inference_timer = QTimer()
        self.demo_inference_timer.timeout.connect(self.simulate_inference_data)
        self.demo_inference_timer.start(2000)  # 0.5 Hz demo predictions
        
        # Demo prediction counter
        self.demo_prediction_count = 0
        
    def set_hardware_devices(self, rng_device, eeg_device):
        """Set hardware device references"""
        self.rng_device = rng_device
        self.eeg_device = eeg_device
        
        # Set up data callbacks if devices are available
        if self.rng_device:
            self.rng_device.add_data_callback(self.on_rng_data)
            
        if self.eeg_device:
            self.eeg_device.add_data_callback(self.on_eeg_data)
            
    def mode_changed(self, index):
        """Handle mode selection change"""
        self.stop_inference()
        mode = index + 1
        
        # Create new inference engine for the selected mode
        config = InferenceConfig(
            mode=mode,
            prediction_rate=self.rate_slider.value(),
            sequence_length=100,
            enable_streaming=self.streaming_check.isChecked(),
            stream_port=getattr(self, 'stream_port', 8765),
            stream_id=getattr(self, 'stream_id', f"inference_mode_{mode}")
        )
        
        engine = InferenceEngine(config)
        engine.add_prediction_callback(self.on_prediction)
        
        self.inference_engines[mode] = engine
        self.current_engine = engine
        
        self.model_info_label.setText("No model selected")
    
    def populate_model_list(self):
        """Populate the model selector with available trained models"""
        try:
            # Clear current items except the first one
            self.model_selector.clear()
            self.model_selector.addItem("Select a trained model...")
            
            # Get available models from model manager
            model_manager = ModelManager()
            available_models = model_manager.get_available_models()
            
            if not available_models:
                self.model_selector.addItem("No trained models found")
                self.update_status("No trained models found", "info")
                return
            
            # Add models to dropdown
            for model_name, metadata in available_models.items():
                # Create display text with model info
                display_text = f"{model_name} ({metadata.variant_config.architecture}, {metadata.variant_config.framework})"
                self.model_selector.addItem(display_text, model_name)  # Store model name as data
            
            self.update_status(f"Found {len(available_models)} trained models", "info")
                
        except Exception as e:
            print(f"Error populating model list: {e}")
            self.model_selector.clear()
            self.model_selector.addItem("Error loading models")
            self.update_status(f"Error loading models: {e}", "error")
    
    def model_changed(self, index):
        """Handle model selection change"""
        if index == 0:  # "Select a trained model..." option
            self.model_info_label.setText("No model selected")
            self.start_button.setEnabled(False)
            self.update_status("⚠️ No model selected", "error")
            return
            
        # Get selected model name from item data
        model_name = self.model_selector.itemData(index)
        if not model_name:
            return
            
        # Show loading status
        self.update_status(f"🔄 Loading model: {model_name}...", "info")
            
        try:
            # Get model metadata
            model_manager = ModelManager()
            metadata = model_manager.get_model_by_name(model_name)
            
            if not metadata:
                self.model_info_label.setText("Error: Model not found")
                return
            
            # Update model info display
            info_text = f"""
            <b>Model:</b> {metadata.variant_config.name}<br>
            <b>Framework:</b> {metadata.variant_config.framework} ({metadata.framework_version})<br>
            <b>Architecture:</b> {metadata.variant_config.architecture}<br>
            <b>Features:</b> {', '.join(metadata.variant_config.input_features)}<br>
            <b>Hidden Size:</b> {metadata.variant_config.hidden_size}<br>
            <b>Layers:</b> {metadata.variant_config.num_layers}<br>
            <b>Validation Loss:</b> {metadata.final_val_loss:.6f}<br>
            <b>Training Date:</b> {metadata.training_time[:19]}<br>
            <b>GPU Used:</b> {'Yes' if metadata.gpu_used else 'No'}
            """.strip()
            
            self.model_info_label.setText(info_text)
            
            # Create inference engine for this model
            self.load_selected_model(model_name, metadata)
            
        except Exception as e:
            self.model_info_label.setText(f"Error loading model info: {e}")
            self.update_status(f"Error: {e}", "error")
            print(f"Error in model_changed: {e}")
    
    def load_selected_model(self, model_name: str, metadata):
        """Load the selected model for inference"""
        try:
            # Stop any current inference
            self.stop_inference()
            self.update_status("Loading model...", "info")
            
            # Create inference config based on model features
            config = MultiModelInferenceConfig(
                sequence_length=metadata.variant_config.sequence_length,
                enable_gpu=metadata.variant_config.use_gpu,
                max_models=1  # For single model inference
            )
            
            # Create advanced inference engine
            advanced_engine = AdvancedInferenceEngine(config)
            
            # Load the specific model
            success = advanced_engine.load_model(model_name)
            
            if success:
                self.current_advanced_engine = advanced_engine
                self.current_model_name = model_name
                self.start_button.setEnabled(True)
                self.update_status(f"Model '{model_name}' loaded successfully ✓", "ready")
                print(f"Successfully loaded model: {model_name}")
            else:
                self.start_button.setEnabled(False)
                self.update_status("Failed to load model", "error")
                
        except Exception as e:
            self.start_button.setEnabled(False)
            self.update_status(f"Error loading model: {e}", "error")
            print(f"Error loading model {model_name}: {e}")
            
    def load_model(self):
        """Legacy method - now handled by model_changed"""
        pass  # This method is now replaced by the model dropdown
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status label with appropriate styling"""
        colors = {
            "ready": {"bg": "#e8f5e8", "color": "#2e7d32", "border": "#c8e6c9"},
            "running": {"bg": "#e3f2fd", "color": "#1565c0", "border": "#90caf9"},
            "error": {"bg": "#ffebee", "color": "#c62828", "border": "#ef9a9a"},
            "stopped": {"bg": "#f3f4f6", "color": "#4b5563", "border": "#d1d5db"},
            "info": {"bg": "#f8f9fa", "color": "#495057", "border": "#dee2e6"}
        }
        
        style_info = colors.get(status_type, colors["info"])
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {style_info['bg']};
                color: {style_info['color']};
                padding: 8px;
                border: 2px solid {style_info['border']};
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }}
        """)
            
    def toggle_recursive_mode(self, enabled: bool):
        """Toggle recursive inference mode on/off"""
        self.recursive_mode = enabled
        self.recursive_depth_slider.setEnabled(enabled)
        
        if enabled:
            print("🔄 RECURSIVE CONSCIOUSNESS MODE ACTIVATED")
            print(f"🌀 Recursion depth: {self.recursive_depth}")
            
            # Initialize recursive layers
            self.recursive_layers = []
            for layer in range(self.recursive_depth):
                layer_data = {
                    'id': layer,
                    'buffer': queue.Queue(maxsize=100),
                    'last_output': None,
                    'feedback_weight': 0.5 ** (layer + 1)  # Exponential decay
                }
                self.recursive_layers.append(layer_data)
                print(f"  📊 Layer {layer}: weight={layer_data['feedback_weight']:.3f}")
                
            self.update_status("🌀 Recursive mode enabled - Consciousness feedback loops active", "running")
        else:
            print("🛑 RECURSIVE MODE DEACTIVATED")
            self.recursive_layers = []
            self.feedback_buffer = queue.Queue(maxsize=1000)
            self.update_status("📊 Standard inference mode", "ready")
            
    def rebuild_recursive_layers(self):
        """Rebuild recursive layers when depth changes"""
        if self.recursive_mode:
            print(f"🔄 Rebuilding recursive layers with depth {self.recursive_depth}")
            
            # Preserve existing layer data where possible
            old_layers = self.recursive_layers.copy()
            self.recursive_layers = []
            
            for layer in range(self.recursive_depth):
                # Try to preserve existing layer data
                existing_data = None
                if layer < len(old_layers):
                    existing_data = old_layers[layer]
                
                layer_data = {
                    'id': layer,
                    'buffer': existing_data['buffer'] if existing_data else queue.Queue(maxsize=100),
                    'last_output': existing_data['last_output'] if existing_data else None,
                    'feedback_weight': 0.5 ** (layer + 1)  # Exponential decay
                }
                self.recursive_layers.append(layer_data)
                print(f"  📊 Layer {layer}: weight={layer_data['feedback_weight']:.3f}")
            
    def start_inference(self):
        """Start inference"""
        # Use new advanced inference engine if available
        if self.current_advanced_engine:
            try:
                # For now, just enable the interface - real streaming would need hardware integration
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.update_status(f"🔄 Running inference with {self.current_model_name}", "running")
                print(f"Started inference with model: {self.current_model_name}")
                
                # TODO: Integrate with hardware data streams for real-time inference
                
            except Exception as e:
                print(f"Error starting inference: {e}")
                self.update_status(f"❌ Error: {e}", "error")
                return
        
        # Fallback to legacy engine
        elif self.current_engine:
            self.current_engine.start_inference()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_status("🔄 Running inference (legacy mode)", "running")
        else:
            self.update_status("⚠️ No model loaded", "error")
        
    def stop_inference(self):
        """Stop inference"""
        # Stop advanced inference engine
        if self.current_advanced_engine:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_status("⏹️ Stopped", "stopped")
            print("Stopped advanced inference")
            
        # Stop legacy engine
        elif self.current_engine:
            self.current_engine.stop_inference()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_status("⏹️ Stopped", "stopped")
        
    def on_rng_data(self, rng_sample):
        """Handle new RNG data"""
        if self.current_engine:
            self.current_engine.add_rng_data(rng_sample.normalized)
            
        # Update visualization
        self.input_viz.add_rng_data(rng_sample.normalized[:4])
        
        # Feed RNG data to mystical field
        if hasattr(self, 'field_widget') and self.field_widget:
            # Convert normalized RNG to bytes for the field
            rng_bytes = bytes([int(x * 255) for x in rng_sample.normalized[:8]])
            self.field_widget.add_rng_data(rng_bytes)
        
    def on_eeg_data(self, eeg_sample):
        """Handle new EEG data"""
        if self.current_engine:
            self.current_engine.add_eeg_data(eeg_sample.channels)
            
        # Update visualization with full channel data
        self.input_viz.add_eeg_data(eeg_sample.channels)
        
    def on_prediction(self, result: PredictionResult):
        """Handle prediction result"""
        # Update confidence
        confidence = int(result.confidence * 100)
        self.confidence_bar.setValue(confidence)
        
        # Update color display
        if len(result.colors) >= 4:
            r, g, b, a = result.colors[:4]
            color_str = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f})"
            self.color_display.setStyleSheet(f"background-color: {color_str}")
            
        # Update dial system
        if len(result.dials) > 0:
            # Convert dial predictions to dial positions
            dial_positions = self._convert_dial_predictions(result.dials)
            self.dial_system.update_animation(0.1, input_data=result.input_data)
            
        # Auto-generate if enabled
        if self.auto_generate_check.isChecked():
            self.generate_from_prediction(result)
            
    def _convert_dial_predictions(self, dial_predictions: np.ndarray) -> Dict:
        """Convert ML dial predictions to dial system format"""
        # This would convert the model output to dial positions
        # Simplified implementation
        positions = {}
        
        # Assume dial_predictions contains flattened dial data
        max_dials = 8
        features_per_dial = 4  # x, y, rotation, radius
        
        for i in range(min(max_dials, len(dial_predictions) // features_per_dial)):
            base_idx = i * features_per_dial
            positions[i] = {
                'center': (dial_predictions[base_idx], dial_predictions[base_idx + 1], 0),
                'rotation': dial_predictions[base_idx + 2],
                'radius': dial_predictions[base_idx + 3]
            }
            
        return positions
        
    def generate_single(self):
        """Generate a single output"""
        print("=== GENERATE SINGLE BUTTON CLICKED ===")
        
        if not self.generated_canvas:
            print("ERROR: No generated canvas available!")
            return
            
        print(f"Canvas available: {self.generated_canvas.canvas_width}x{self.generated_canvas.canvas_height}")
        
        if not self.current_engine:
            print("No current engine - creating test prediction")
            # Create a test prediction for demonstration
            self.generate_test_output()
            return
            
        # Use current data buffers to make a prediction
        try:
            print("Attempting to get RNG sequence...")
            rng_data = self.current_engine._get_rng_sequence()
            eeg_data = self.current_engine._get_eeg_sequence() if self.current_engine.config.mode == 2 else None
            
            if rng_data is not None:
                print("RNG data found, making prediction...")
                prediction = self.current_engine._make_prediction()
                if prediction:
                    print("Generated prediction from engine")
                    self.generate_from_prediction(prediction)
                else:
                    print("Engine prediction failed - generating test output")
                    self.generate_test_output()
            else:
                print("No RNG data - generating test output")
                self.generate_test_output()
        except Exception as e:
            print(f"Error generating output: {e}")
            print("Falling back to test output...")
            self.generate_test_output()
    
    def generate_test_output(self):
        """Generate a test output for demonstration purposes"""
        print("=== GENERATING TEST OUTPUT ===")
        import random
        
        if not self.generated_canvas:
            print("ERROR: No canvas for test output!")
            return
        
        # Create a fake prediction result for testing
        fake_result = type('PredictionResult', (), {
            'colors': [random.random(), random.random(), random.random(), 0.8],
            'curves': [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1.0), 0.5, 0.0],
            'confidence': random.random(),
            'dials': [],
            'input_data': None
        })()
        
        print(f"Test colors: {fake_result.colors}")
        print(f"Test curves: {fake_result.curves}")
        
        self.generate_from_prediction(fake_result)
                
    def generate_from_prediction(self, result: PredictionResult):
        """Generate visual output from prediction result"""
        if not self.generated_canvas:
            return
            
        # Convert prediction to drawing actions
        if len(result.colors) >= 4 and len(result.curves) >= 5:
            # Extract color
            r, g, b, a = result.colors[:4]
            color = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            
            # Extract curve parameters
            x, y, brush_size, pressure, _ = result.curves[:5]
            
            # Normalize position to canvas (convert from [-1,1] to canvas coordinates)
            canvas_width = self.generated_canvas.canvas_width
            canvas_height = self.generated_canvas.canvas_height
            canvas_x = int((x + 1) * canvas_width / 2)   # Convert [-1,1] to [0, width]
            canvas_y = int((y + 1) * canvas_height / 2)  # Convert [-1,1] to [0, height]
            
            # Ensure coordinates are within bounds
            canvas_x = max(0, min(canvas_width - 1, canvas_x))
            canvas_y = max(0, min(canvas_height - 1, canvas_y))
            
            # Set drawing parameters
            self.generated_canvas.set_color(color)
            brush_size_val = max(1, min(50, int(abs(brush_size) * 20)))  # Scale and clamp brush size
            self.generated_canvas.set_brush_size(brush_size_val)
            
            # Draw directly on the canvas pixmap
            painter = QPainter(self.generated_canvas.pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Create pen with the prediction color and size
            pen = QPen(color, brush_size_val, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            
            # Generate a small artistic stroke
            import random
            start_point = QPoint(canvas_x, canvas_y)
            
            # Create a curved stroke with multiple points
            points = [start_point]
            for i in range(1, 8):  # 8-point stroke
                # Create organic curve movement
                angle = (i / 7.0) * math.pi * 2 + random.uniform(-0.5, 0.5)
                radius = brush_size_val * (2 + i * 0.3)
                
                new_x = canvas_x + int(radius * math.cos(angle))
                new_y = canvas_y + int(radius * math.sin(angle))
                
                # Keep within bounds
                new_x = max(0, min(canvas_width - 1, new_x))
                new_y = max(0, min(canvas_height - 1, new_y))
                
                points.append(QPoint(new_x, new_y))
            
            # Draw connected lines between points for smooth stroke
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
            
            painter.end()
            
            # Update the canvas display
            self.generated_canvas.update()
            
            print(f"Generated drawing: color=({r:.2f},{g:.2f},{b:.2f},{a:.2f}), pos=({canvas_x},{canvas_y}), size={brush_size_val}")
        else:
            print(f"Insufficient prediction data: colors={len(result.colors)}, curves={len(result.curves)}")
                
    def clear_generated(self):
        """Clear generated artwork"""
        if self.generated_canvas:
            self.generated_canvas.clear_canvas()
            
    def save_generated(self):
        """Save generated artwork"""
        # In practice, implement file saving
        print("Save functionality would be implemented here")
        
    def test_generate_immediate(self):
        """Immediate test generation for debugging"""
        print("TEST GENERATE BUTTON CLICKED!")
        
        if not self.generated_canvas:
            print("ERROR: No generated canvas found!")
            return
            
        print(f"Canvas found: {self.generated_canvas.canvas_width}x{self.generated_canvas.canvas_height}")
        
        # Draw a simple test pattern
        painter = QPainter(self.generated_canvas.pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw a red circle
        red_pen = QPen(QColor(255, 0, 0), 5)
        painter.setPen(red_pen)
        painter.drawEllipse(100, 100, 100, 100)
        
        # Draw a blue line
        blue_pen = QPen(QColor(0, 0, 255), 3)
        painter.setPen(blue_pen)
        painter.drawLine(QPoint(50, 50), QPoint(250, 250))
        
        # Draw a green rectangle
        green_pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(green_pen)
        painter.drawRect(300, 150, 100, 75)
        
        painter.end()
        
        # Force update
        self.generated_canvas.update()
        print("Test drawing completed - should see red circle, blue line, and green rectangle")
        
    def update_ui(self):
        """Update UI elements"""
        if self.current_engine:
            stats = self.current_engine.get_statistics()
            self.prediction_count_label.setText(f"Predictions: {stats['total_predictions']}")
            
    def update_statistics(self):
        """Update statistics display"""
        if not self.current_engine:
            return
            
        stats = self.current_engine.get_statistics()
        
        stats_text = f"""
INFERENCE STATISTICS
====================

Mode: {self.current_engine.config.mode}
Total Predictions: {stats['total_predictions']}
Average Prediction Time: {stats['average_prediction_time']:.4f} sec
Current Rate: {stats['current_rate']:.2f} Hz

Buffer Status:
  RNG Buffer: {stats['buffer_sizes']['rng']} samples
  EEG Buffer: {stats['buffer_sizes']['eeg']} samples

Hardware Status:
  RNG Device: {'Connected' if self.rng_device and self.rng_device.is_connected else 'Disconnected'}
  EEG Device: {'Connected' if self.eeg_device and self.eeg_device.is_connected else 'Disconnected'}
        """
        
        self.stats_text.setPlainText(stats_text)
        
    def configure_streaming(self, port: int, stream_id: str):
        """Configure streaming parameters and auto-start engine"""
        # Enable streaming checkbox
        self.streaming_check.setChecked(True)
        
        # Store parameters for when engine is created
        self.stream_port = port
        self.stream_id = stream_id
        
        print(f"Configured inference streaming: port={port}, id={stream_id}")
        
        # Auto-start an inference engine for streaming
        try:
            # Determine mode based on available devices
            has_eeg = self.eeg_device and hasattr(self.eeg_device, 'is_connected') and self.eeg_device.is_connected
            if has_eeg:
                mode = 2  # RNG + EEG
            else:
                mode = 1  # RNG only
                
            print(f"Auto-starting inference engine: mode={mode}, has_eeg={has_eeg}")
            
            # Set mode selector and create engine
            self.mode_selector.setCurrentIndex(mode - 1)  # Modes are 1-based, index is 0-based
            self.mode_changed(mode - 1)
            
            # Auto-load and start the engine
            self.load_model()
            self.start_inference()
            
            print(f"Auto-started inference engine for streaming (mode {mode})")
            
        except Exception as e:
            print(f"Failed to auto-start inference engine: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_mystical_field(self):
        """Reset the mystical field to initial state"""
        if hasattr(self, 'field_widget') and self.field_widget:
            # Clear existing particles and recreate initial ones
            self.field_widget.field.particles.clear()
            self.field_widget.field._create_initial_particles()
            self.update_status("🌟 Mystical field reset", "info")
            print("Mystical field reset to initial state")
    
    def update_field_intensity(self, value):
        """Update mystical field intensity based on slider"""
        if hasattr(self, 'field_widget') and self.field_widget:
            # Scale intensity (value is 10-200, normalize to 0.1-2.0)
            intensity = value / 100.0
            
            # Update field physics parameters
            field = self.field_widget.field
            field.jitter_strength = 0.5 * intensity
            field.attraction_strength = 0.8 * intensity
            field.repulsion_strength = 0.3 * intensity
            field.gravity_strength = 0.02 * intensity
            
            print(f"Updated mystical field intensity: {intensity:.2f}")
    
    def update_particle_count(self):
        """Update particle count display"""
        if hasattr(self, 'field_widget') and self.field_widget and hasattr(self, 'particle_count_label'):
            count = len(self.field_widget.field.particles)
            self.particle_count_label.setText(f"Particles: {count}")
    
    def simulate_field_data(self):
        """Generate synthetic RNG data to keep mystical field active during testing"""
        if hasattr(self, 'field_widget') and self.field_widget:
            # Generate pseudo-random data with some interesting patterns
            import random
            import time
            
            # Use time-based patterns for more interesting field behavior
            t = time.time()
            
            # Generate 8 bytes with various patterns
            synthetic_rng = []
            for i in range(8):
                # Mix of random and deterministic patterns
                base = int((math.sin(t * 0.5 + i) + 1) * 127.5)  # Sine wave component
                noise = random.randint(-30, 30)  # Random noise
                chaos = int(random.random() * 255)  # Pure chaos
                
                # Blend different components
                value = int((base * 0.4 + (chaos * 0.4) + ((base + noise) * 0.2)) % 256)
                # Ensure value is in valid byte range
                value = max(0, min(255, value))
                synthetic_rng.append(value)
            
            # Feed to field
            rng_bytes = bytes(synthetic_rng)
            self.field_widget.add_rng_data(rng_bytes)
            
    def simulate_inference_data(self):
        """Generate simulated inference data for demonstration with recursive support"""
        import random
        import time
        import numpy as np
        
        # Increment demo prediction counter
        self.demo_prediction_count += 1
        
        # Update prediction count display
        self.prediction_count_label.setText(f"Predictions: {self.demo_prediction_count}")
        
        # Generate base prediction result
        t = time.time()
        
        # Base confidence without recursion
        base_confidence = 0.5 + 0.4 * math.sin(t * 0.3)
        
        # Base colors
        base_r = 0.5 + 0.5 * math.sin(t * 0.7)
        base_g = 0.5 + 0.5 * math.sin(t * 0.5 + 2.094)
        base_b = 0.5 + 0.5 * math.sin(t * 0.3 + 4.188)
        base_a = 0.8 + 0.2 * math.sin(t * 1.1)
        
        # Apply recursive processing if enabled
        if self.recursive_mode and len(self.recursive_layers) > 0:
            r, g, b, a, confidence = self.process_recursive_layers(base_r, base_g, base_b, base_a, base_confidence, t)
        else:
            r, g, b, a, confidence = base_r, base_g, base_b, base_a, base_confidence
        
        # Update displays
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update color display
        color_str = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f})"
        self.color_display.setStyleSheet(f"background-color: {color_str}; border: 2px solid #444; border-radius: 5px;")
        
        # Update dial display with recursive information
        if self.recursive_mode:
            dial_text = f"3D Dial Visualization\n🌀 RECURSIVE MODE 🌀\nDepth: {self.recursive_depth}\nLayers: {len(self.recursive_layers)}\nRotation: {(t * 30) % 360:.1f}°"
        else:
            dial_text = f"3D Dial Visualization\n⚙️ Active ⚙️\nRotation: {(t * 30) % 360:.1f}°\nRadius: {0.5 + 0.3 * math.sin(t * 0.8):.2f}"
        self.dial_display.setText(dial_text)
        
        # Update input visualization with synthetic data
        if hasattr(self, 'input_viz') and self.input_viz:
            # Generate synthetic RNG and EEG data
            rng_data = [random.randint(0, 255) for _ in range(32)]
            eeg_data = {
                'alpha': 8.5 + 2 * math.sin(t * 0.4),
                'beta': 15.2 + 3 * math.sin(t * 0.6 + 1),
                'gamma': 35.8 + 5 * math.sin(t * 0.8 + 2),
                'delta': 2.1 + 1 * math.sin(t * 0.2 + 3),
                'theta': 6.3 + 1.5 * math.sin(t * 0.5 + 4)
            }
            
            # Update the visualization widget
            try:
                self.input_viz.add_rng_data([val / 255.0 for val in rng_data[:4]])  # Normalize to 0-1
                self.input_viz.add_eeg_data(eeg_data)
                self.input_viz.update()
            except AttributeError:
                # If the methods don't exist, just refresh the widget
                self.input_viz.update()
        
        # Auto-generate if enabled
        if hasattr(self, 'auto_generate_check') and self.auto_generate_check.isChecked():
            # Create a fake prediction result and generate artwork
            fake_result = type('PredictionResult', (), {
                'colors': [r, g, b, a],
                'curves': [
                    math.sin(t * 0.3) * 0.8,  # x position
                    math.cos(t * 0.4) * 0.6,  # y position  
                    0.3 + 0.2 * math.sin(t * 0.7),  # brush size
                    0.8,  # pressure
                    0.0   # extra
                ],
                'confidence': confidence,
                'dials': [],
                'input_data': None,
                'recursive_layer': len(self.recursive_layers) if self.recursive_mode else 0
            })()
            
            # Only auto-generate occasionally to avoid cluttering
            if self.demo_prediction_count % 3 == 0:  # Every 3rd prediction
                self.generate_from_prediction(fake_result)
        
        mode_str = f" (recursive depth {self.recursive_depth})" if self.recursive_mode else ""
        print(f"Demo prediction #{self.demo_prediction_count}{mode_str}: confidence={confidence:.2f}, color=({r:.2f},{g:.2f},{b:.2f},{a:.2f})")
    
    def process_recursive_layers(self, base_r, base_g, base_b, base_a, base_confidence, t):
        """Process data through recursive consciousness layers"""
        current_output = {
            'r': base_r, 'g': base_g, 'b': base_b, 'a': base_a,
            'confidence': base_confidence,
            'timestamp': t
        }
        
        # Process through each recursive layer
        for layer_data in self.recursive_layers:
            layer_id = layer_data['id']
            weight = layer_data['feedback_weight']
            
            # Get previous output from this layer for feedback
            prev_output = layer_data['last_output']
            
            if prev_output is not None:
                # Apply feedback from previous cycle
                feedback_strength = 0.3 * weight
                current_output['r'] = (1 - feedback_strength) * current_output['r'] + feedback_strength * prev_output['r']
                current_output['g'] = (1 - feedback_strength) * current_output['g'] + feedback_strength * prev_output['g']
                current_output['b'] = (1 - feedback_strength) * current_output['b'] + feedback_strength * prev_output['b']
                current_output['confidence'] = (1 - feedback_strength) * current_output['confidence'] + feedback_strength * prev_output['confidence']
                
                # Add some recursive instability for consciousness effect
                instability = 0.1 * weight * math.sin(t * (layer_id + 1) * 2.3)
                current_output['r'] = max(0, min(1, current_output['r'] + instability))
                current_output['g'] = max(0, min(1, current_output['g'] + instability * 0.7))
                current_output['b'] = max(0, min(1, current_output['b'] + instability * 1.3))
            
            # Store this layer's output for next cycle
            layer_data['last_output'] = current_output.copy()
            
            # Add to buffer for potential logging
            try:
                layer_data['buffer'].put_nowait({
                    'layer': layer_id,
                    'output': current_output.copy(),
                    'timestamp': t
                })
            except queue.Full:
                # Remove oldest item if buffer is full
                try:
                    layer_data['buffer'].get_nowait()
                    layer_data['buffer'].put_nowait({
                        'layer': layer_id,
                        'output': current_output.copy(),
                        'timestamp': t
                    })
                except queue.Empty:
                    pass
        
        # Return processed values
        return (current_output['r'], current_output['g'], current_output['b'], 
                current_output['a'], current_output['confidence'])
    
    def initialize_demo_mode(self):
        """Initialize the interface with demo data to show functionality"""
        # Set initial color display
        self.color_display.setStyleSheet("background-color: rgba(157, 78, 221, 0.8); border: 2px solid #444; border-radius: 5px;")
        
        # Set initial dial display
        self.dial_display.setText("3D Dial Visualization\n⚙️ Demo Mode ⚙️\nInitializing...\nReady for predictions")
        
        # Set initial confidence
        self.confidence_bar.setValue(50)
        
        # Set initial status
        self.update_status("🔄 Demo mode active - Live simulation running", "running")
        
        print("Demo mode initialized - Real-time simulation active!")
    
    def update_ui(self):
        """Update UI elements"""
        if self.current_engine:
            stats = self.current_engine.get_statistics()
            self.prediction_count_label.setText(f"Predictions: {stats['total_predictions']}")
        
        # Update particle count for mystical field
        self.update_particle_count()


def create_inference_app(rng_device=None, eeg_device=None, enable_streaming=False, stream_port=8765, stream_id="inference_1"):
    """
    Create and return the inference application
    
    Args:
        rng_device: TrueRNG device instance  
        eeg_device: EEG device instance
        enable_streaming: Enable network streaming
        stream_port: Port for streaming server
        stream_id: ID for the stream
        
    Returns:
        Tuple of (app, window)
    """
    if not PYQT_AVAILABLE:
        raise ImportError("PyQt5 is required for the inference GUI")
        
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        
    window = InferenceWindow()
    if rng_device or eeg_device:
        window.set_hardware_devices(rng_device, eeg_device)
        
    # Configure streaming if requested
    if enable_streaming:
        window.configure_streaming(stream_port, stream_id)
        
    return app, window


# Example usage
if __name__ == "__main__":
    if PYQT_AVAILABLE:
        app, window = create_inference_app()
        window.show()
        app.exec_()
    else:
        print("PyQt5 not available. Install with: pip install PyQt5")