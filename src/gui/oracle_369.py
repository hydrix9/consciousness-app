"""
369 Oracle Mode: Sophisticated Consciousness Oracle System
This module implements the three-layer consciousness oracle with vector mathematics
and ChatGPT integration for consciousness interpretation.
"""

import sys
import time
import math
import numpy as np
import threading
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QLineEdit, QProgressBar,
    QGroupBox, QGridLayout, QSplitter, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

try:
    from .painting_interface import PaintCanvas, DrawingAction, ConsciousnessVector
except ImportError:
    from gui.painting_interface import PaintCanvas, DrawingAction, ConsciousnessVector

try:
    from ml.oracle_network import Oracle369NetworkManager, create_oracle_network_manager
except ImportError:
    from src.ml.oracle_network import Oracle369NetworkManager, create_oracle_network_manager

try:
    from ml.inference_interface import PredictionResult
except ImportError:
    from src.ml.inference_interface import PredictionResult


class ConsciousnessState(Enum):
    """Quantum-inspired consciousness states for the 369 Oracle"""
    GROUND = "ground"           # Base consciousness state
    COHERENT = "coherent"       # Synchronized across layers
    ENTANGLED = "entangled"     # Non-local consciousness correlations
    SUPERPOSITION = "superposition"  # Multiple states simultaneously
    RESONANCE = "resonance"     # Frequency-locked patterns
    TRANSCENDENT = "transcendent"    # Beyond dimensional boundaries


@dataclass
class ConsciousnessReading:
    """Consciousness data from a single reading cycle"""
    timestamp: float
    layer_1_vector: ConsciousnessVector
    layer_2_vector: ConsciousnessVector
    layer_3_vector: ConsciousnessVector
    unified_vector: ConsciousnessVector
    state: ConsciousnessState
    coherence_index: float      # 0.0 to 1.0
    resonance_frequency: float  # Hz
    quantum_phase: float        # 0 to 2π
    interpretation: str = ""    # ChatGPT analysis


class ConsciousnessMathEngine:
    """Advanced mathematics for consciousness vector operations"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio - fundamental to 369
        self.pi = math.pi
        self.e = math.e
        
    def calculate_consciousness_vector(self, actions: List[DrawingAction], 
                                    timespan: float = 9.0) -> ConsciousnessVector:
        """Convert drawing actions into consciousness vector over 9-second window"""
        if not actions:
            return ConsciousnessVector(0, 0, 0, 0, time.time())
            
        # Filter actions within timespan
        current_time = time.time()
        recent_actions = [a for a in actions if current_time - a.timestamp <= timespan]
        
        if not recent_actions:
            return ConsciousnessVector(0, 0, 0, 0, current_time)
            
        # Calculate base vectors from action properties
        x_component = sum(a.position[0] for a in recent_actions) / len(recent_actions)
        y_component = sum(a.position[1] for a in recent_actions) / len(recent_actions)
        
        # Z-component from pressure and brush dynamics
        z_component = sum(a.pressure * a.brush_size for a in recent_actions) / len(recent_actions)
        
        # W-component (consciousness intensity) from action frequency and color variance
        time_density = len(recent_actions) / timespan
        color_variance = self._calculate_color_entropy(recent_actions)
        w_component = time_density * color_variance * self.phi
        
        return ConsciousnessVector(x_component, y_component, z_component, w_component, current_time)
    
    def _calculate_color_entropy(self, actions: List[DrawingAction]) -> float:
        """Calculate entropy of color usage - measure of consciousness creativity"""
        if not actions:
            return 0.0
            
        # Get unique colors and their frequencies
        color_counts = {}
        for action in actions:
            color_key = tuple(action.color[:3])  # RGB only
            color_counts[color_key] = color_counts.get(color_key, 0) + 1
            
        total_actions = len(actions)
        entropy = 0.0
        
        for count in color_counts.values():
            probability = count / total_actions
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    def calculate_unified_vector(self, layer1: ConsciousnessVector, 
                               layer2: ConsciousnessVector, 
                               layer3: ConsciousnessVector) -> ConsciousnessVector:
        """Combine three layer vectors using 369 sacred geometry"""
        
        # 369 weighting: Layer 1 = 3, Layer 2 = 6, Layer 3 = 9
        weight_1, weight_2, weight_3 = 3, 6, 9
        total_weight = weight_1 + weight_2 + weight_3
        
        # Weighted average with golden ratio modulation
        unified_x = (layer1.x * weight_1 + layer2.x * weight_2 + layer3.x * weight_3) / total_weight
        unified_y = (layer1.y * weight_1 + layer2.y * weight_2 + layer3.y * weight_3) / total_weight
        unified_z = (layer1.z * weight_1 + layer2.z * weight_2 + layer3.z * weight_3) / total_weight
        
        # W-component uses harmonic resonance
        w_resonance = math.sqrt(layer1.w**2 + layer2.w**2 + layer3.w**2) * self.phi
        
        return ConsciousnessVector(unified_x, unified_y, unified_z, w_resonance, time.time())
    
    def determine_consciousness_state(self, reading: ConsciousnessReading) -> ConsciousnessState:
        """Determine quantum consciousness state from vector analysis"""
        
        # Calculate coherence between layers
        coherence = self.calculate_coherence(
            reading.layer_1_vector, 
            reading.layer_2_vector, 
            reading.layer_3_vector
        )
        
        # Calculate magnitude ratios for state determination
        unified_magnitude = reading.unified_vector.magnitude()
        layer_magnitudes = [
            reading.layer_1_vector.magnitude(),
            reading.layer_2_vector.magnitude(),
            reading.layer_3_vector.magnitude()
        ]
        
        avg_layer_magnitude = sum(layer_magnitudes) / 3
        
        # State determination logic
        if coherence > 0.9:
            return ConsciousnessState.COHERENT
        elif coherence > 0.7 and unified_magnitude > avg_layer_magnitude * self.phi:
            return ConsciousnessState.RESONANCE
        elif any(mag > unified_magnitude * 1.5 for mag in layer_magnitudes):
            return ConsciousnessState.SUPERPOSITION
        elif unified_magnitude > avg_layer_magnitude * 2:
            return ConsciousnessState.TRANSCENDENT
        elif coherence > 0.5:
            return ConsciousnessState.ENTANGLED
        else:
            return ConsciousnessState.GROUND
    
    def calculate_coherence(self, v1: ConsciousnessVector, 
                          v2: ConsciousnessVector, 
                          v3: ConsciousnessVector) -> float:
        """Calculate coherence index between consciousness vectors"""
        
        # Dot products for angular similarities
        dot_12 = v1.dot_product(v2)
        dot_13 = v1.dot_product(v3)
        dot_23 = v2.dot_product(v3)
        
        # Magnitude products
        mag_12 = v1.magnitude() * v2.magnitude()
        mag_13 = v1.magnitude() * v3.magnitude()
        mag_23 = v2.magnitude() * v3.magnitude()
        
        # Cosine similarities (avoiding division by zero)
        cos_12 = dot_12 / mag_12 if mag_12 > 0 else 0
        cos_13 = dot_13 / mag_13 if mag_13 > 0 else 0
        cos_23 = dot_23 / mag_23 if mag_23 > 0 else 0
        
        # Average coherence
        return (abs(cos_12) + abs(cos_13) + abs(cos_23)) / 3
    
    def calculate_resonance_frequency(self, unified_vector: ConsciousnessVector) -> float:
        """Calculate consciousness resonance frequency in Hz"""
        
        # Base frequency from W-component (consciousness intensity)
        base_freq = unified_vector.w * 0.1  # Scale to reasonable Hz range
        
        # Modulate with golden ratio and vector magnitude
        magnitude = unified_vector.magnitude()
        modulated_freq = base_freq * (1 + magnitude / (100 * self.phi))
        
        # Keep in meaningful frequency range (0.1 Hz to 100 Hz)
        return max(0.1, min(100.0, modulated_freq))


class Oracle369Interface(QMainWindow):
    """Main interface for the 369 Oracle consciousness system"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("369 Oracle - Consciousness Interpretation System")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Core components
        self.math_engine = ConsciousnessMathEngine()
        self.consciousness_readings: List[ConsciousnessReading] = []
        self.recording_active = False
        self.oracle_question = ""
        
        # Three painting canvases for the three layers
        self.layer_canvases: Dict[int, PaintCanvas] = {}
        
        # Network manager for receiving inference streams
        self.network_manager: Optional[Oracle369NetworkManager] = None
        self.network_enabled = False
        self.inference_data: Dict[int, List[PredictionResult]] = {1: [], 2: [], 3: []}
        
        # Setup UI
        self.setup_ui()
        
        # Data collection timer (runs every 100ms for smooth data capture)
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.collect_consciousness_data)
        
        # Oracle reading timer (every 3 seconds for 3-6-9 rhythm)
        self.oracle_timer = QTimer()
        self.oracle_timer.timeout.connect(self.perform_oracle_reading)
        
        # Initialize network connections
        self.initialize_network_connections()
        
    def setup_ui(self):
        """Create the 369 Oracle interface"""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Splitter for main content
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Left side: Three consciousness layers
        layers_widget = self.create_layers_widget()
        content_splitter.addWidget(layers_widget)
        
        # Right side: Oracle interface and data
        oracle_widget = self.create_oracle_widget()
        content_splitter.addWidget(oracle_widget)
        
        # Set splitter proportions (70% layers, 30% oracle)
        content_splitter.setSizes([1120, 480])
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel with oracle controls"""
        
        control_group = QGroupBox("369 Oracle Controls")
        layout = QHBoxLayout(control_group)
        
        # Question input
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your consciousness question for the Oracle...")
        layout.addWidget(QLabel("Question:"))
        layout.addWidget(self.question_input)
        
        # Oracle button
        self.oracle_btn = QPushButton("🔮 Consult Oracle (9 seconds)")
        self.oracle_btn.clicked.connect(self.start_oracle_session)
        layout.addWidget(self.oracle_btn)
        
        # Network mode controls
        self.network_btn = QPushButton("🌐 Enable Network Mode")
        self.network_btn.clicked.connect(self.toggle_network_mode)
        layout.addWidget(self.network_btn)
        
        # Status display
        self.status_label = QLabel("Ready - Local Mode")
        layout.addWidget(self.status_label)
        
        # Progress bar for oracle sessions
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return control_group
    
    def create_layers_widget(self) -> QWidget:
        """Create the three consciousness layer painting interfaces"""
        
        layers_widget = QWidget()
        layout = QVBoxLayout(layers_widget)
        
        # Create three painting interfaces
        for layer_num in [1, 2, 3]:
            # Layer group
            layer_group = QGroupBox(f"Consciousness Layer {layer_num}")
            layer_layout = QVBoxLayout(layer_group)
            
            # Create painting canvas for this layer
            painting_canvas = PaintCanvas(400, 300)  # Smaller size for oracle
            painting_canvas.set_consciousness_layer(layer_num)
            
            # Set mystical layer-specific colors to match main interface
            layer_colors = {
                1: QColor(157, 78, 221),    # #9D4EDD - Ethereal Violet
                2: QColor(61, 90, 128),     # #3D5A80 - Cosmic Blue  
                3: QColor(75, 0, 130)       # #4B0082 - Shadow Indigo
            }
            painting_canvas.set_color(layer_colors[layer_num])
            
            layer_layout.addWidget(painting_canvas)
            layout.addWidget(layer_group)
            
            # Store reference
            self.layer_canvases[layer_num] = painting_canvas
        
        return layers_widget
    
    def create_oracle_widget(self) -> QWidget:
        """Create the oracle interpretation and data display widget"""
        
        oracle_widget = QWidget()
        layout = QVBoxLayout(oracle_widget)
        
        # Oracle interpretation display
        interpretation_group = QGroupBox("Oracle Interpretation")
        interp_layout = QVBoxLayout(interpretation_group)
        
        self.interpretation_display = QTextEdit()
        self.interpretation_display.setReadOnly(True)
        self.interpretation_display.setFont(QFont("Arial", 11))
        interp_layout.addWidget(self.interpretation_display)
        
        layout.addWidget(interpretation_group)
        
        # Consciousness data display
        data_group = QGroupBox("Consciousness Mathematics")
        data_layout = QVBoxLayout(data_group)
        
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setFont(QFont("Courier", 9))
        data_layout.addWidget(self.data_display)
        
        layout.addWidget(data_group)
        
        # Set proportions (60% interpretation, 40% data)
        interpretation_group.setMinimumHeight(300)
        data_group.setMinimumHeight(200)
        
        return oracle_widget
    
    def start_oracle_session(self):
        """Start a 9-second oracle consciousness reading session"""
        
        if self.recording_active:
            return
            
        self.oracle_question = self.question_input.text().strip()
        if not self.oracle_question:
            self.status_label.setText("Please enter a question first")
            return
            
        # Clear previous data
        self.consciousness_readings.clear()
        for canvas in self.layer_canvases.values():
            canvas.clear_canvas()
            
        # Start recording
        self.recording_active = True
        self.oracle_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.status_label.setText("🔮 Oracle session active - Express your consciousness...")
        
        # Start data collection and oracle timers
        self.data_timer.start(100)  # Collect data every 100ms
        self.oracle_timer.start(3000)  # Oracle reading every 3 seconds
        
        # Stop after 9 seconds
        QTimer.singleShot(9000, self.complete_oracle_session)
        
        # Update progress bar
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(90)  # Update every 90ms for smooth progress
        
    def update_progress(self):
        """Update the progress bar during oracle session"""
        current_value = self.progress_bar.value()
        if current_value < 100:
            self.progress_bar.setValue(current_value + 1)
        else:
            self.progress_timer.stop()
            
    def collect_consciousness_data(self):
        """Collect consciousness data from all three layers"""
        
        if not self.recording_active:
            return
            
        # Calculate consciousness vectors for each layer
        layer_vectors = {}
        for layer_num, canvas in self.layer_canvases.items():
            actions = canvas.drawing_actions
            vector = self.math_engine.calculate_consciousness_vector(actions)
            layer_vectors[layer_num] = vector
            
        # Calculate unified vector
        if len(layer_vectors) == 3:
            unified_vector = self.math_engine.calculate_unified_vector(
                layer_vectors[1], layer_vectors[2], layer_vectors[3]
            )
            
            # Create consciousness reading
            reading = ConsciousnessReading(
                timestamp=time.time(),
                layer_1_vector=layer_vectors[1],
                layer_2_vector=layer_vectors[2], 
                layer_3_vector=layer_vectors[3],
                unified_vector=unified_vector,
                state=ConsciousnessState.GROUND,  # Will be calculated
                coherence_index=0.0,
                resonance_frequency=0.0,
                quantum_phase=0.0
            )
            
            # Calculate advanced properties
            reading.state = self.math_engine.determine_consciousness_state(reading)
            reading.coherence_index = self.math_engine.calculate_coherence(
                reading.layer_1_vector, reading.layer_2_vector, reading.layer_3_vector
            )
            reading.resonance_frequency = self.math_engine.calculate_resonance_frequency(
                reading.unified_vector
            )
            reading.quantum_phase = (reading.unified_vector.w * math.pi) % (2 * math.pi)
            
            self.consciousness_readings.append(reading)
            
    def perform_oracle_reading(self):
        """Perform an oracle reading with current consciousness data"""
        
        if not self.consciousness_readings:
            return
            
        latest_reading = self.consciousness_readings[-1]
        
        # Display mathematical data
        self.display_consciousness_data(latest_reading)
        
    def display_consciousness_data(self, reading: ConsciousnessReading):
        """Display consciousness mathematical data"""
        
        data_text = f"""
CONSCIOUSNESS MATHEMATICS - {time.strftime('%H:%M:%S')}
{'='*50}

LAYER VECTORS:
Layer 1 (Primary): ({reading.layer_1_vector.x:.2f}, {reading.layer_1_vector.y:.2f}, {reading.layer_1_vector.z:.2f}, {reading.layer_1_vector.w:.2f})
Layer 2 (Subconscious): ({reading.layer_2_vector.x:.2f}, {reading.layer_2_vector.y:.2f}, {reading.layer_2_vector.z:.2f}, {reading.layer_2_vector.w:.2f})
Layer 3 (Universal): ({reading.layer_3_vector.x:.2f}, {reading.layer_3_vector.y:.2f}, {reading.layer_3_vector.z:.2f}, {reading.layer_3_vector.w:.2f})

UNIFIED CONSCIOUSNESS:
Vector: ({reading.unified_vector.x:.2f}, {reading.unified_vector.y:.2f}, {reading.unified_vector.z:.2f}, {reading.unified_vector.w:.2f})
Magnitude: {reading.unified_vector.magnitude():.3f}

QUANTUM PROPERTIES:
State: {reading.state.value.upper()}
Coherence Index: {reading.coherence_index:.3f}
Resonance Frequency: {reading.resonance_frequency:.2f} Hz
Quantum Phase: {reading.quantum_phase:.3f} rad

369 SACRED RATIOS:
Golden Ratio φ: {self.math_engine.phi:.6f}
Layer Harmony: {(reading.coherence_index * self.math_engine.phi):.3f}
Consciousness Intensity: {reading.unified_vector.w:.3f}
"""
        
        self.data_display.setPlainText(data_text)
        
    def complete_oracle_session(self):
        """Complete the oracle session and generate interpretation"""
        
        # Stop timers
        self.data_timer.stop()
        self.oracle_timer.stop()
        self.progress_timer.stop()
        
        self.recording_active = False
        self.oracle_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_label.setText("Processing oracle interpretation...")
        
        # Generate consciousness summary for ChatGPT
        consciousness_summary = self.generate_consciousness_summary()
        
        # TODO: Integrate with ChatGPT API for interpretation
        # For now, display a sophisticated mock interpretation
        interpretation = self.generate_mock_interpretation(consciousness_summary)
        
        self.interpretation_display.setPlainText(interpretation)
        self.status_label.setText("Oracle interpretation complete")
        
    def generate_consciousness_summary(self) -> str:
        """Generate a summary of consciousness data for ChatGPT interpretation"""
        
        if not self.consciousness_readings:
            return "No consciousness data collected"
            
        # Analyze the complete 9-second session
        total_readings = len(self.consciousness_readings)
        
        # Calculate session statistics
        states = [r.state for r in self.consciousness_readings]
        avg_coherence = sum(r.coherence_index for r in self.consciousness_readings) / total_readings
        avg_frequency = sum(r.resonance_frequency for r in self.consciousness_readings) / total_readings
        
        # Dominant state
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
        dominant_state = max(state_counts, key=state_counts.get)
        
        # Consciousness evolution over time
        final_reading = self.consciousness_readings[-1]
        
        summary = f"""
Question: {self.oracle_question}

Consciousness Session Analysis (9 seconds, {total_readings} readings):
- Dominant State: {dominant_state.value}
- Average Coherence: {avg_coherence:.3f}
- Average Resonance: {avg_frequency:.2f} Hz
- Final Vector Magnitude: {final_reading.unified_vector.magnitude():.3f}
- Final Consciousness Intensity: {final_reading.unified_vector.w:.3f}

This consciousness data represents the querent's intuitive response patterns while contemplating their question through creative expression across three consciousness layers.
"""
        
        return summary
        
    def generate_mock_interpretation(self, consciousness_summary: str) -> str:
        """Generate a sophisticated mock interpretation (placeholder for ChatGPT)"""
        
        # This would be replaced with actual ChatGPT API integration
        interpretation = f"""
🔮 ORACLE INTERPRETATION - 369 CONSCIOUSNESS ANALYSIS
{'='*60}

{consciousness_summary}

INTERPRETATION:
The consciousness patterns revealed through your 9-second meditation show a profound alignment with universal energies. The mathematical harmony of your creative expression indicates...

[This would be generated by ChatGPT based on the consciousness mathematics and your specific question]

The 369 oracle speaks through the language of vectors and quantum consciousness states, translating your intuitive wisdom into coherent guidance.

KEY INSIGHTS:
• Your consciousness resonance suggests...
• The layer coherence indicates...  
• The quantum state reveals...

Remember: The oracle reflects your own inner wisdom, amplified through consciousness mathematics.
"""
        
        return interpretation

    def initialize_network_connections(self):
        """Initialize network connections to inference streams"""
        try:
            self.network_manager = create_oracle_network_manager()
            self.network_manager.add_layer_callback(self.on_inference_prediction)
            self.network_manager.add_status_callback(self.on_network_status)
            
            # Don't auto-start - let user enable manually
            print("Oracle network manager initialized. Use 'Enable Network Mode' to connect.")
            
        except Exception as e:
            print(f"Warning: Could not initialize network manager: {e}")
            
    def toggle_network_mode(self):
        """Toggle between local and network mode"""
        if self.network_enabled:
            self.disable_network_mode()
            self.network_btn.setText("🌐 Enable Network Mode")
            self.status_label.setText("Ready - Local Mode")
        else:
            self.enable_network_mode()
            self.network_btn.setText("🔌 Disable Network Mode") 
            self.status_label.setText("Network Mode - Connecting...")

    def enable_network_mode(self):
        """Enable network mode to receive inference streams"""
        if not self.network_manager:
            self.status_label.setText("Network manager not available")
            return
            
        try:
            self.network_manager.start_all_connections()
            self.network_enabled = True
            self.status_label.setText("Network Mode - Connected to inference streams")
            print("Oracle network mode enabled. Connecting to inference streams...")
            print("Expected connections:")
            print("  Layer 1 (Primary): localhost:8765")
            print("  Layer 2 (Subconscious): localhost:8766") 
            print("  Layer 3 (Universal): localhost:8767")
            
        except Exception as e:
            self.status_label.setText(f"Network error: {e}")
            print(f"Error enabling network mode: {e}")
            
    def disable_network_mode(self):
        """Disable network mode"""
        if self.network_manager:
            try:
                self.network_manager.stop_all_connections()
                self.network_enabled = False
                self.status_label.setText("Ready - Local Mode")
                print("Oracle network mode disabled")
                
            except Exception as e:
                self.status_label.setText(f"Network error: {e}")
                print(f"Error disabling network mode: {e}")
                
    def on_inference_prediction(self, layer: int, prediction: PredictionResult):
        """Handle incoming prediction from inference stream"""
        if not self.network_enabled:
            return
            
        # Store prediction data for the layer
        self.inference_data[layer].append(prediction)
        
        # Limit buffer size
        if len(self.inference_data[layer]) > 50:
            self.inference_data[layer] = self.inference_data[layer][-50:]
            
        # Update the Oracle interface with live data
        self.update_layer_from_inference(layer, prediction)
        
    def on_network_status(self, message: str):
        """Handle network status updates"""
        print(f"Oracle Network: {message}")
        
    def update_layer_from_inference(self, layer: int, prediction: PredictionResult):
        """Update Oracle layer with inference prediction data"""
        if layer not in self.layer_canvases:
            return
            
        canvas = self.layer_canvases[layer]
        
        # Convert prediction to drawing actions
        if len(prediction.colors) >= 4 and len(prediction.curves) >= 5:
            # Extract color
            r, g, b, a = prediction.colors[:4]
            color = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            
            # Extract position and drawing parameters
            x, y, brush_size, pressure, _ = prediction.curves[:5]
            
            # Normalize to canvas coordinates
            canvas_x = int((x + 1) * canvas.width() / 2)  # [-1,1] -> [0,width]
            canvas_y = int((y + 1) * canvas.height() / 2)  # [-1,1] -> [0,height]
            
            # Apply to canvas
            canvas.set_color(color)
            canvas.set_brush_size(max(1, int(brush_size * 20)))
            
            # Create a small drawing action to represent the inference
            # This could be expanded to create more sophisticated visualizations
            
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network connection status"""
        if not self.network_manager:
            return {"enabled": False, "message": "Network manager not available"}
            
        status = self.network_manager.get_connection_status()
        stats = self.network_manager.get_all_statistics()
        
        return {
            "enabled": self.network_enabled,
            "connections": status,
            "statistics": stats,
            "all_connected": self.network_manager.is_all_connected()
        }


def main():
    """Run the 369 Oracle application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the oracle interface
    oracle = Oracle369Interface()
    oracle.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()