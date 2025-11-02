"""
Creative Painting Interface

This module provides a GUI for painting swashes of color and drawing lines
while capturing real-time data from hardware devices.
"""

import sys
import time
import math
import logging
import os
import json
import random
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QSlider, QColorDialog,
                                QGroupBox, QGridLayout, QTextEdit, QSplitter, QTabWidget,
                                QCheckBox, QComboBox)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
    from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QPainterPath, 
                            QPixmap, QFont, QPalette)
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Create dummy classes for type hints
    class QMainWindow: pass
    class QWidget: pass
    class QThread: pass
    class pyqtSignal: pass

import numpy as np
import math

# Import mystical field components for merged mode
try:
    from gui.field_visualization import create_mystical_field_widget
    MYSTICAL_FIELD_AVAILABLE = True
except ImportError:
    MYSTICAL_FIELD_AVAILABLE = False
    def create_mystical_field_widget():
        return None

# Import 3D curve/dial system for visualization
try:
    from src.utils.curve_3d import InterlockingDialSystem, LineToCircleConverter
    DIAL_SYSTEM_AVAILABLE = True
except ImportError:
    try:
        from utils.curve_3d import InterlockingDialSystem, LineToCircleConverter
        DIAL_SYSTEM_AVAILABLE = True
    except ImportError:
        DIAL_SYSTEM_AVAILABLE = False


@dataclass
class DrawingAction:
    """Data structure for a drawing action"""
    timestamp: float
    action_type: str  # 'stroke_start', 'stroke_continue', 'stroke_end', 'color_change', 'layer_change', 'dimension_change'
    position: Tuple[float, float]
    color: Tuple[int, int, int, int]  # RGBA
    brush_size: int
    pressure: float
    consciousness_layer: int  # Layer 1, 2, or 3 (369 system)
    pocket_dimension: int  # Current pocket dimension for consciousness navigation
    rng_data: Optional[List[float]] = None
    eeg_data: Optional[dict] = None
    metadata: Optional[dict] = None  # Additional data for special actions like dimension changes


@dataclass 
class ConsciousnessVector:
    """4D consciousness vector for quantum consciousness mathematics"""
    x: float  # Spatial consciousness X
    y: float  # Spatial consciousness Y  
    z: float  # Depth consciousness Z
    w: float  # Consciousness intensity/quantum state
    timestamp: float
    
    def magnitude(self) -> float:
        """Calculate the magnitude of the consciousness vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def dot_product(self, other: 'ConsciousnessVector') -> float:
        """Calculate dot product with another consciousness vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def normalize(self) -> 'ConsciousnessVector':
        """Return normalized consciousness vector"""
        mag = self.magnitude()
        if mag == 0:
            return ConsciousnessVector(0, 0, 0, 0, self.timestamp)
        return ConsciousnessVector(
            self.x / mag, self.y / mag, self.z / mag, self.w / mag, self.timestamp
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'x': self.x,
            'y': self.y, 
            'z': self.z,
            'w': self.w,
            'magnitude': self.magnitude(),
            'timestamp': self.timestamp
        }


class PaintCanvas(QWidget):
    """Custom widget for painting canvas"""
    
    def __init__(self, width: int = 800, height: int = 600):
        super().__init__()
        
        self.canvas_width = width
        self.canvas_height = height
        self.setFixedSize(width, height)
        
        # Canvas state - create pixmap with mystical dark background
        self.pixmap = QPixmap(width, height)
        self.pixmap.fill(QColor(20, 20, 30))  # Dark mystical background (#14141E)
        
        # Dial visualization overlay (white brush strokes showing 3D geometry)
        self.dial_overlay = QPixmap(width, height)
        self.dial_overlay.fill(Qt.transparent)
        
        # Dial system for 3D curve conversion
        self.dial_system = InterlockingDialSystem() if DIAL_SYSTEM_AVAILABLE else None
        self.show_dial_visualization = False  # Toggle for showing dial overlay
        self.current_stroke_points = []  # Points in the current stroke
        
        # Ensure the widget supports alpha blending
        self.setAttribute(Qt.WA_TranslucentBackground, False)  # We want opaque background
        self.setAutoFillBackground(True)
        
        # Drawing state with mystical default color
        self.drawing = False
        self.last_point = QPoint()
        self.current_color = QColor(157, 78, 221, 255)  # Default to Ethereal Violet #9D4EDD
        self.brush_size = 10
        self.brush_opacity = 255  # 0-255 (0 = transparent, 255 = opaque)
        self.current_pressure = 1.0
        
        # Consciousness layer system (369)
        self.consciousness_layer = 1  # Current active layer (1, 2, or 3)
        
        # Drawing history
        self.drawing_actions: List[DrawingAction] = []
        
        # Callbacks for real-time data
        self.action_callbacks: List[Callable[[DrawingAction], None]] = []
        
        # Mouse tracking
        self.setMouseTracking(True)
        
    def set_color(self, color: QColor):
        """Set current drawing color"""
        self.current_color = color
        # Preserve current opacity
        self.current_color.setAlpha(self.brush_opacity)
        
    def set_brush_size(self, size: int):
        """Set current brush size"""
        self.brush_size = size
        
    def set_brush_opacity(self, opacity: int):
        """Set current brush opacity (0-255)"""
        self.brush_opacity = opacity
        self.current_color.setAlpha(opacity)
        logging.debug(f"Brush opacity set to {opacity} (alpha: {self.current_color.alpha()})")
        
    def set_consciousness_layer(self, layer: int):
        """Set the active consciousness layer (1, 2, or 3)"""
        self.consciousness_layer = layer
    
    def set_dial_visualization(self, enabled: bool):
        """Toggle dial visualization overlay"""
        self.show_dial_visualization = enabled
        if not enabled:
            # Clear dial overlay
            self.dial_overlay.fill(Qt.transparent)
            if self.dial_system and DIAL_SYSTEM_AVAILABLE:
                self.dial_system.clear_all_dials()
        self.update()
    
    def _render_dial_visualization(self):
        """Render the current dial system as white curves on the overlay"""
        if not self.dial_system or not DIAL_SYSTEM_AVAILABLE:
            return
        
        # Clear overlay
        self.dial_overlay.fill(Qt.transparent)
        
        painter = QPainter(self.dial_overlay)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # White pen for dial visualization
        dial_pen = QPen(QColor(255, 255, 255, 200), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(dial_pen)
        
        # Draw each dial's curve segments
        for dial_id, dial in self.dial_system.dials.items():
            for segment in dial.segments:
                # Draw the 3D curve as a white path
                if len(segment.points) < 2:
                    continue
                
                # Convert 3D points to 2D for display
                prev_point = None
                for point_3d in segment.points:
                    # Project 3D to 2D (simple orthographic projection ignoring Z)
                    x = int(point_3d.x)
                    y = int(point_3d.y)
                    
                    if prev_point:
                        painter.drawLine(prev_point[0], prev_point[1], x, y)
                    
                    prev_point = (x, y)
            
            # Draw dial center and radius indicators
            center_x = int(dial.center.x)
            center_y = int(dial.center.y)
            radius = int(dial.radius)
            
            # Draw a subtle circle to show the dial boundary
            circle_pen = QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine)
            painter.setPen(circle_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
            
            # Draw center point
            center_pen = QPen(QColor(255, 255, 255, 255), 4)
            painter.setPen(center_pen)
            painter.drawPoint(center_x, center_y)
        
        painter.end()
        
    def clear_canvas(self):
        """Clear the canvas with mystical dark background"""
        self.pixmap.fill(QColor(20, 20, 30))  # Dark mystical background
        self.dial_overlay.fill(Qt.transparent)
        self.drawing_actions.clear()
        self.current_stroke_points.clear()
        if self.dial_system and DIAL_SYSTEM_AVAILABLE:
            self.dial_system.clear_all_dials()
        self.update()
        
    def add_action_callback(self, callback: Callable[[DrawingAction], None]):
        """Add callback for drawing actions"""
        self.action_callbacks.append(callback)
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
            # Start new stroke for dial system
            self.current_stroke_points = [(event.pos().x(), event.pos().y())]
            
            # Draw initial dot to show opacity immediately
            painter = QPainter(self.pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Create a fresh color with the current opacity
            paint_color = QColor(self.current_color.red(), self.current_color.green(), 
                               self.current_color.blue(), self.brush_opacity)
            
            # Debug: Log current color and alpha for initial dot
            logging.debug(f"Initial dot with color: RGBA({paint_color.red()}, {paint_color.green()}, {paint_color.blue()}, {paint_color.alpha()})")
            
            # Set up pen for drawing
            pen = QPen(paint_color, self.brush_size, 
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            
            # Draw a small dot (just a point) at the click location
            painter.drawPoint(event.pos())
            painter.end()
            
            self.update()
            
            # Record action
            action = DrawingAction(
                timestamp=time.time(),
                action_type='stroke_start',
                position=(event.pos().x(), event.pos().y()),
                color=(paint_color.red(), paint_color.green(), 
                      paint_color.blue(), paint_color.alpha()),
                brush_size=self.brush_size,
                pressure=self.current_pressure,
                consciousness_layer=self.consciousness_layer,
                pocket_dimension=self.parent().pocket_dimension if hasattr(self.parent(), 'pocket_dimension') else 1
            )
            
            self.drawing_actions.append(action)
            self._call_action_callbacks(action)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if event.buttons() & Qt.LeftButton and self.drawing:
            # Add point to current stroke
            self.current_stroke_points.append((event.pos().x(), event.pos().y()))
            
            painter = QPainter(self.pixmap)
            
            # Use SourceOver for proper alpha blending
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Create a fresh color with the current opacity
            paint_color = QColor(self.current_color.red(), self.current_color.green(), 
                               self.current_color.blue(), self.brush_opacity)
            
            # Debug: Log current color and alpha
            logging.debug(f"Drawing with color: RGBA({paint_color.red()}, {paint_color.green()}, {paint_color.blue()}, {paint_color.alpha()})")
            
            # Set up pen with explicit alpha color
            pen = QPen(paint_color, self.brush_size, 
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            
            # Draw line
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            
            # End painter
            painter.end()
            
            # Record action
            action = DrawingAction(
                timestamp=time.time(),
                action_type='stroke_continue',
                position=(event.pos().x(), event.pos().y()),
                color=(paint_color.red(), paint_color.green(), 
                      paint_color.blue(), paint_color.alpha()),
                brush_size=self.brush_size,
                pressure=self.current_pressure,
                consciousness_layer=self.consciousness_layer,
                pocket_dimension=self.parent().pocket_dimension if hasattr(self.parent(), 'pocket_dimension') else 1
            )
            
            self.drawing_actions.append(action)
            self._call_action_callbacks(action)
            
            self.update()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            
            # Add final point to stroke
            self.current_stroke_points.append((event.pos().x(), event.pos().y()))
            
            # Convert stroke to dial geometry if visualization is enabled
            if self.show_dial_visualization and self.dial_system and DIAL_SYSTEM_AVAILABLE:
                if len(self.current_stroke_points) >= 3:  # Need at least 3 points
                    # Convert to dial system
                    color_tuple = (
                        self.current_color.red() / 255.0,
                        self.current_color.green() / 255.0,
                        self.current_color.blue() / 255.0,
                        self.current_color.alpha() / 255.0
                    )
                    
                    dial_id = self.dial_system.add_stroke(
                        self.current_stroke_points,
                        (self.canvas_width, self.canvas_height),
                        color_tuple
                    )
                    
                    if dial_id is not None:
                        logging.debug(f"Created dial {dial_id} from stroke with {len(self.current_stroke_points)} points")
                        # Render the updated dial visualization
                        self._render_dial_visualization()
            
            # Clear stroke points
            self.current_stroke_points = []
            
            # Record action
            action = DrawingAction(
                timestamp=time.time(),
                action_type='stroke_end',
                position=(event.pos().x(), event.pos().y()),
                color=(self.current_color.red(), self.current_color.green(), 
                      self.current_color.blue(), self.current_color.alpha()),
                brush_size=self.brush_size,
                pressure=self.current_pressure,
                consciousness_layer=self.consciousness_layer,
                pocket_dimension=self.parent().pocket_dimension if hasattr(self.parent(), 'pocket_dimension') else 1
            )
            
            self.drawing_actions.append(action)
            self._call_action_callbacks(action)
            
    def paintEvent(self, event):
        """Handle paint events"""
        painter = QPainter(self)
        
        # Draw the main canvas
        painter.drawPixmap(self.rect(), self.pixmap, self.pixmap.rect())
        
        # Draw the dial visualization overlay if enabled
        if self.show_dial_visualization:
            painter.drawPixmap(self.rect(), self.dial_overlay, self.dial_overlay.rect())
        
    def _call_action_callbacks(self, action: DrawingAction):
        """Call all registered action callbacks"""
        for callback in self.action_callbacks:
            try:
                callback(action)
            except Exception as e:
                print(f"Error in action callback: {e}")


class DataVisualizationWidget(QWidget):
    """Widget for visualizing real-time data streams"""
    
    def __init__(self, width: int = 400, height: int = 300):
        super().__init__()
        
        self.setFixedSize(width, height)
        
        # Data buffers
        self.rng_data: List[float] = []
        self.eeg_channels: dict = {
            'AF3': [], 'AF4': [], 'F3': [], 'F4': [], 'F7': [], 'F8': [],
            'FC5': [], 'FC6': [], 'P7': [], 'P8': [], 'T7': [], 'T8': [], 'O1': [], 'O2': []
        }
        self.max_points = 100  # Reduced for better performance
        
        # Visualization state
        self.rng_color = QColor(255, 100, 100)  # Light red for RNG
        self.eeg_colors = [
            QColor(0, 255, 255),   # Cyan
            QColor(255, 255, 0),   # Yellow
            QColor(255, 0, 255),   # Magenta
            QColor(0, 255, 0),     # Green
            QColor(255, 128, 0),   # Orange
            QColor(128, 0, 255),   # Purple
            QColor(255, 192, 203), # Pink
            QColor(0, 255, 128),   # Spring Green
            QColor(128, 255, 0),   # Lime
            QColor(255, 0, 128),   # Deep Pink
            QColor(0, 128, 255),   # Sky Blue
            QColor(255, 255, 128), # Light Yellow
            QColor(128, 255, 255), # Light Cyan
            QColor(255, 128, 255)  # Light Magenta
        ]
        
        # EEG activity indicators
        self.eeg_activity_levels = {channel: 0.0 for channel in self.eeg_channels.keys()}
        self.last_eeg_update = 0
        
    def add_rng_data(self, values: List[float]):
        """Add RNG data points"""
        self.rng_data.extend(values)
        if len(self.rng_data) > self.max_points:
            self.rng_data = self.rng_data[-self.max_points:]
        self.update()
        
    def add_eeg_data(self, eeg_sample: dict):
        """Add EEG data from all channels"""
        current_time = time.time()
        self.last_eeg_update = current_time
        
        for channel, value in eeg_sample.items():
            if channel in self.eeg_channels:
                self.eeg_channels[channel].append(value)
                if len(self.eeg_channels[channel]) > self.max_points:
                    self.eeg_channels[channel] = self.eeg_channels[channel][-self.max_points:]
                
                # Calculate activity level (RMS of recent values)
                recent_values = self.eeg_channels[channel][-10:]  # Last 10 samples
                if recent_values:
                    rms = np.sqrt(np.mean([v**2 for v in recent_values]))
                    self.eeg_activity_levels[channel] = min(rms / 100.0, 1.0)  # Normalize to 0-1
        
        self.update()
        
    def paintEvent(self, event):
        """Paint the data visualization"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))  # Dark background
        
        width = self.width()
        height = self.height()
        
        # Calculate panel heights
        rng_height = height // 3
        eeg_height = height * 2 // 3
        
        # Draw RNG data
        if self.rng_data:
            painter.setPen(QPen(self.rng_color, 2))
            self._draw_sparkline(painter, self.rng_data, 10, 0, width - 20, rng_height - 20, 
                               min_val=0, max_val=1)
            
            # RNG label and status
            painter.setPen(QPen(Qt.white, 1))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(10, 15, "RNG Data Stream")
            
            # Draw RNG activity indicator
            activity = len(self.rng_data) / self.max_points
            indicator_color = QColor(int(255 * activity), int(255 * (1 - activity)), 0)
            painter.fillRect(width - 30, 5, 20, 10, indicator_color)
                    
        # Draw EEG sparklines
        eeg_start_y = rng_height
        channel_names = list(self.eeg_channels.keys())
        channels_per_row = 7
        channel_width = width // channels_per_row
        channel_height = eeg_height // 2
        
        for i, (channel, data) in enumerate(self.eeg_channels.items()):
            if not data:
                continue
                
            row = i // channels_per_row
            col = i % channels_per_row
            
            x_start = col * channel_width + 5
            y_start = eeg_start_y + row * channel_height + 20
            channel_w = channel_width - 10
            channel_h = channel_height - 30
            
            # Draw channel background
            bg_color = QColor(40, 40, 40)
            painter.fillRect(x_start, y_start, channel_w, channel_h, bg_color)
            
            # Draw sparkline
            color_index = i % len(self.eeg_colors)
            painter.setPen(QPen(self.eeg_colors[color_index], 1))
            
            if len(data) > 1:
                data_range = max(abs(max(data)), abs(min(data))) if data else 1
                self._draw_sparkline(painter, data, x_start + 2, y_start + 2, 
                                   channel_w - 4, channel_h - 4, 
                                   min_val=-data_range, max_val=data_range)
            
            # Draw channel label
            painter.setPen(QPen(Qt.white, 1))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(x_start + 2, y_start - 5, channel)
            
            # Draw activity indicator
            activity = self.eeg_activity_levels.get(channel, 0)
            indicator_color = QColor(int(255 * activity), int(255 * (1 - activity)), 50)
            painter.fillRect(x_start + channel_w - 8, y_start - 8, 6, 6, indicator_color)
        
        # EEG connection status
        painter.setPen(QPen(Qt.white, 1))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(10, eeg_start_y + 15, "EEG Channel Activity")
        
        # Connection indicator
        current_time = time.time()
        connection_active = (current_time - self.last_eeg_update) < 2.0  # 2 second timeout
        status_color = QColor(0, 255, 0) if connection_active else QColor(255, 0, 0)
        painter.fillRect(width - 30, eeg_start_y + 5, 20, 10, status_color)
        
    def _draw_sparkline(self, painter, data, x, y, width, height, min_val=None, max_val=None):
        """Draw a sparkline for the given data"""
        if len(data) < 2:
            return
            
        if min_val is None:
            min_val = min(data)
        if max_val is None:
            max_val = max(data)
            
        if max_val == min_val:
            max_val += 0.1  # Avoid division by zero
            
        points = []
        for i, value in enumerate(data):
            px = x + int((i / (len(data) - 1)) * width)
            py = y + height - int(((value - min_val) / (max_val - min_val)) * height)
            points.append(QPoint(px, py))
        
        # Draw the line
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])
    
    def update_displays(self):
        """Update data visualization displays - called by main interface"""
        # Trigger a repaint to update the visualization
        self.update()
        
        # Optional: Add any other display update logic here
        # This method is called regularly by the main interface timer


class ConsciousnessMainWindow(QMainWindow):
    """Main window for the consciousness data generation app"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Consciousness Data Generator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Hardware interfaces (will be injected)
        self.rng_device = None
        self.eeg_device = None
        
        # Data collection
        self.session_active = False
        self.session_start_time = 0
        
        # Merged mode variables for mystical field and inference
        self.field_widget = None
        self.inference_mode_enabled = False
        self.recursive_session_count = 0
        
        self.setup_ui()
        self.setup_timers()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for color selection"""
        key = event.key()
        
        # Check for numeric keys 1-8 for color selection
        if key >= Qt.Key_1 and key <= Qt.Key_8:
            key_num = str(key - Qt.Key_1 + 1)  # Convert to "1", "2", etc.
            key_name = f"key_{key_num}"
            
            if key_name in self.color_buttons:
                color_hex, btn = self.color_buttons[key_name]
                self.set_quick_color(color_hex)
                logging.info(f"🎨 Color changed via keyboard: {key_num} -> {color_hex}")
                
                # Visual feedback - briefly highlight the button
                original_style = btn.styleSheet()
                btn.setStyleSheet(original_style + "border: 3px solid yellow;")
                QTimer.singleShot(200, lambda: btn.setStyleSheet(original_style))
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        
        # Center panel - Canvas
        center_panel = self.create_canvas_panel()
        
        # Right panel - Data visualization
        right_panel = self.create_data_panel()
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800, 400])
        
        main_layout.addWidget(splitter)
        
        # Initialize opacity to match slider
        self.opacity_changed(255)  # Set initial opacity to 100%
        
    def create_control_panel(self) -> QWidget:
        """Create the control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Session controls
        session_group = QGroupBox("Session Control")
        session_layout = QVBoxLayout(session_group)
        
        self.start_button = QPushButton("Start Session")
        self.start_button.clicked.connect(self.start_session)
        self.stop_button = QPushButton("Stop Session")
        self.stop_button.clicked.connect(self.stop_session)
        self.stop_button.setEnabled(False)
        
        session_layout.addWidget(self.start_button)
        session_layout.addWidget(self.stop_button)
        
        # Drawing controls
        drawing_group = QGroupBox("Drawing Controls")
        drawing_layout = QVBoxLayout(drawing_group)
        
        # Quick color palette
        color_palette_layout = QVBoxLayout()
        color_palette_layout.addWidget(QLabel("Colors (Press 1-8):"))
        
        colors_row_layout = QHBoxLayout()
        self.color_buttons = {}
        # Simplified palette: White, Purple, Blue, Cyan, Gold, Red, Indigo, Green
        colors = [
            ("Pure White", "#FFFFFF", "1"),          # White for dial visualization
            ("Ethereal Violet", "#9D4EDD", "2"),     # Deep mystical purple
            ("Cosmic Blue", "#4169E1", "3"),         # Royal blue (simplified from dark blue)
            ("Astral Cyan", "#00F5FF", "4"),         # Bright ethereal cyan
            ("Mystic Gold", "#FFD700", "5"),         # Golden divine light
            ("Crimson Aura", "#DC143C", "6"),        # Deep occult red
            ("Shadow Indigo", "#4B0082", "7"),       # Dark mystical indigo
            ("Plasma Green", "#39FF14", "8")         # Electric supernatural green
        ]
        
        for idx, (name, hex_color, key) in enumerate(colors):
            btn = QPushButton(key)  # Show key number on button
            btn.setFixedSize(40, 30)
            # Add glow effect to buttons
            # Special styling for white button (dark border for visibility)
            border_color = "#1a1a1a" if hex_color != "#FFFFFF" else "#666666"
            text_color = "#000000" if hex_color == "#FFFFFF" else "#FFFFFF"
            btn.setStyleSheet(f"""
                background-color: {hex_color}; 
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 3px;
                box-shadow: 0 0 10px {hex_color}40;
                font-weight: bold;
            """)
            btn.clicked.connect(lambda checked, color=hex_color: self.set_quick_color(color))
            btn.setToolTip(f"{name} ({hex_color}) - Press {key}")
            colors_row_layout.addWidget(btn)
            self.color_buttons[name.lower().replace(" ", "_")] = btn
            # Store color for keybind access
            self.color_buttons[f"key_{key}"] = (hex_color, btn)
            
        color_palette_layout.addLayout(colors_row_layout)
        
        # Custom color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Custom:"))
        self.color_button = QPushButton("Choose Color")
        self.color_button.clicked.connect(self.choose_color)
        self.color_button.setStyleSheet("""
            background-color: #9D4EDD; 
            color: white; 
            border: 2px solid #1a1a1a;
            border-radius: 3px;
            box-shadow: 0 0 5px #9D4EDD40;
            font-weight: bold;
        """)
        color_layout.addWidget(self.color_button)
        color_palette_layout.addLayout(color_layout)
        
        # Quick brush sizes
        brush_sizes_layout = QVBoxLayout()
        brush_sizes_layout.addWidget(QLabel("Quick Brush Sizes:"))
        
        sizes_row_layout = QHBoxLayout()
        self.brush_size_buttons = {}
        brush_sizes = [5, 10, 20, 35, 50, 100]  # Added 100px brush size
        
        for size in brush_sizes:
            btn = QPushButton(str(size))
            btn.setFixedSize(40, 30)
            btn.clicked.connect(lambda checked, s=size: self.set_quick_brush_size(s))
            btn.setToolTip(f"Brush size: {size}px")
            # Highlight the default size (10)
            if size == 10:
                btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            sizes_row_layout.addWidget(btn)
            self.brush_size_buttons[size] = btn
            
        brush_sizes_layout.addLayout(sizes_row_layout)
        
        # Brush opacity slider
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Brush Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 255)  # Minimum 10 to ensure visibility
        self.opacity_slider.setValue(255)
        self.opacity_slider.valueChanged.connect(self.opacity_changed)
        self.opacity_label = QLabel("100%")
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        
        # Brush size slider (fine control)
        brush_layout = QVBoxLayout()
        brush_layout.addWidget(QLabel("Fine Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self.brush_size_changed)
        self.brush_size_label = QLabel("10")
        brush_layout.addWidget(self.brush_slider)
        brush_layout.addWidget(self.brush_size_label)
        
        # Consciousness Layers (369 System)
        layers_layout = QVBoxLayout()
        layers_layout.addWidget(QLabel("Consciousness Layers:"))
        layers_row_layout = QHBoxLayout()
        
        self.layer_buttons = {}
        self.current_layer = 1
        
        for layer in [1, 2, 3]:
            btn = QPushButton(f"Layer {layer}")
            btn.setFixedSize(60, 35)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, l=layer: self.handle_layer_and_dimension_change(l, force_layer_change=False))
            
            # Mystical layer colors and descriptions
            layer_info = {
                1: {"color": "#9D4EDD", "name": "Ethereal Consciousness", "desc": "Primary mystical awareness"},
                2: {"color": "#3D5A80", "name": "Cosmic Flow", "desc": "Subconscious astral currents"}, 
                3: {"color": "#4B0082", "name": "Shadow Realm", "desc": "Universal occult connection"}
            }
            
            info = layer_info[layer]
            btn.setToolTip(f"Layer {layer}: {info['name']} - {info['desc']}")
            
            # Set layer 1 as default with mystical styling
            if layer == 1:
                btn.setChecked(True)
                btn.setStyleSheet(f"""
                    background-color: {info['color']}; 
                    color: white; 
                    font-weight: bold;
                    border: 2px solid #1a1a1a;
                    border-radius: 3px;
                    box-shadow: 0 0 8px {info['color']}60;
                """)
            else:
                btn.setStyleSheet(f"""
                    background-color: #2a2a2a; 
                    color: {info['color']}; 
                    border: 1px solid {info['color']};
                    border-radius: 3px;
                """)
            
            layers_row_layout.addWidget(btn)
            self.layer_buttons[layer] = btn
            
        layers_layout.addLayout(layers_row_layout)
        
        # Layer info display with mystical theme
        self.layer_info_label = QLabel("Current: Layer 1 (Ethereal Consciousness)")
        self.layer_info_label.setStyleSheet("font-weight: bold; color: #9D4EDD; background-color: #1a1a1a; padding: 5px; border-radius: 3px;")
        layers_layout.addWidget(self.layer_info_label)
        
        # Pocket Dimension System (integrated with layer buttons)
        pocket_layout = QVBoxLayout()
        pocket_layout.addWidget(QLabel("Pocket Dimension:"))
        
        self.pocket_dimension = 1  # Start at dimension 1
        
        # Pocket dimension display
        self.pocket_display = QLabel(f"Dimension: {self.pocket_dimension}")
        self.pocket_display.setStyleSheet("""
            font-weight: bold; 
            color: #FFD700; 
            background-color: #1a1a1a; 
            padding: 8px; 
            border: 2px solid #FFD700; 
            border-radius: 5px;
            text-align: center;
        """)
        pocket_layout.addWidget(self.pocket_display)
        
        # Static net change display (what just happened)
        self.net_change_display = QLabel("Net: --")
        self.net_change_display.setStyleSheet("""
            font-weight: bold; 
            color: #00FF7F; 
            background-color: #0d1a0d; 
            padding: 6px; 
            border: 2px solid #00FF7F; 
            border-radius: 5px;
            text-align: center;
            font-size: 12px;
        """)
        pocket_layout.addWidget(self.net_change_display)
        
        # Pocket dimension info
        self.pocket_info_label = QLabel("Layer 1: buttons 2,3 navigate | Layer 2: buttons 1,3 | Layer 3: buttons 1,2")
        self.pocket_info_label.setStyleSheet("font-size: 10px; color: #888; font-style: italic;")
        pocket_layout.addWidget(self.pocket_info_label)
        
        layers_layout.addLayout(pocket_layout)
        
        # Dial visualization toggle (for --mode generate)
        dial_viz_layout = QHBoxLayout()
        self.dial_viz_checkbox = QCheckBox("Show Interlocking Dials")
        self.dial_viz_checkbox.setToolTip("Show white visualization of 3D dial geometry from drawing strokes")
        self.dial_viz_checkbox.setChecked(False)
        self.dial_viz_checkbox.stateChanged.connect(self.toggle_dial_visualization)
        dial_viz_layout.addWidget(self.dial_viz_checkbox)
        
        # Clear button
        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.clear_canvas)
        
        drawing_layout.addLayout(color_palette_layout)
        drawing_layout.addLayout(brush_sizes_layout)
        drawing_layout.addLayout(opacity_layout)
        drawing_layout.addLayout(brush_layout)
        drawing_layout.addLayout(layers_layout)
        drawing_layout.addLayout(dial_viz_layout)
        drawing_layout.addWidget(self.clear_button)
        
        # Hardware status
        hardware_group = QGroupBox("Hardware Status")
        hardware_layout = QVBoxLayout(hardware_group)
        
        self.rng_status = QLabel("RNG: Disconnected")
        self.eeg_status = QLabel("EEG: Disconnected")
        
        hardware_layout.addWidget(self.rng_status)
        hardware_layout.addWidget(self.eeg_status)
        
        # Data info
        data_group = QGroupBox("Session Data")
        data_layout = QVBoxLayout(data_group)
        
        self.session_time_label = QLabel("Session Time: 00:00")
        self.drawing_actions_label = QLabel("Drawing Actions: 0")
        self.rng_samples_label = QLabel("RNG Samples: 0")
        self.eeg_samples_label = QLabel("EEG Samples: 0")
        
        data_layout.addWidget(self.session_time_label)
        data_layout.addWidget(self.drawing_actions_label)
        data_layout.addWidget(self.rng_samples_label)
        data_layout.addWidget(self.eeg_samples_label)
        
        # Add all groups to panel
        layout.addWidget(session_group)
        layout.addWidget(drawing_group)
        layout.addWidget(hardware_group)
        layout.addWidget(data_group)
        layout.addStretch()
        
        return panel
        
    def create_canvas_panel(self) -> QWidget:
        """Create the canvas panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Canvas
        self.canvas = PaintCanvas(800, 600)
        self.canvas.add_action_callback(self.on_drawing_action)
        
        # Canvas controls
        canvas_controls = QHBoxLayout()
        canvas_controls.addWidget(QLabel("Drawing Canvas"))
        canvas_controls.addStretch()
        
        layout.addLayout(canvas_controls)
        layout.addWidget(self.canvas)
        
        return panel
        
    def create_data_panel(self) -> QWidget:
        """Create the data visualization panel with mystical field"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for data visualization
        tab_widget = QTabWidget()
        
        # Data visualization tab
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        data_layout.addWidget(QLabel("Real-time Data Streams"))
        self.data_viz = DataVisualizationWidget(400, 450)  # Increased height for EEG channels
        data_layout.addWidget(self.data_viz)
        
        # Log display
        data_layout.addWidget(QLabel("Event Log"))
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)  # Reduced to make room for larger viz
        self.log_display.setReadOnly(True)
        data_layout.addWidget(self.log_display)
        
        tab_widget.addTab(data_tab, "🔬 Data Streams")
        
        # Mystical field tab for merged mode
        field_tab = self.create_mystical_field_tab()
        tab_widget.addTab(field_tab, "🔮 Mystical Field")
        
        # Inference control tab for recursive recording
        inference_tab = self.create_inference_control_tab()
        tab_widget.addTab(inference_tab, "🧠 Inference Mode")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def create_mystical_field_tab(self) -> QWidget:
        """Create the mystical field visualization tab for merged mode"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description label
        desc_label = QLabel("🔮 Mystical Field - Physical particle simulation driven by consciousness RNG stream")
        desc_label.setStyleSheet("""
            QLabel {
                color: #9932CC;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: rgba(153, 50, 204, 0.1);
                border: 1px solid #9932CC;
                border-radius: 5px;
                margin-bottom: 8px;
            }
        """)
        layout.addWidget(desc_label)
        
        # Create mystical field widget
        try:
            if MYSTICAL_FIELD_AVAILABLE:
                self.field_widget = create_mystical_field_widget()
                if self.field_widget:
                    # Set size appropriate for the side panel
                    self.field_widget.setFixedSize(380, 300)
                    layout.addWidget(self.field_widget)
                    
                    # Field controls
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
                    
                    # Intensity control
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
                    
                    # Connect intensity slider
                    self.field_intensity_slider.valueChanged.connect(self.update_field_intensity)
                    
                    # Particle count and reset
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
                else:
                    # Fallback if field widget creation fails
                    fallback_label = QLabel("🔮 Field Unavailable\n\nMystical field widget\ncould not be created")
                    fallback_label.setAlignment(Qt.AlignCenter)
                    fallback_label.setStyleSheet("""
                        QLabel {
                            color: #FF6347; 
                            font-size: 12px;
                            background-color: rgba(255, 99, 71, 0.1);
                            border: 1px solid #FF6347;
                            border-radius: 8px;
                            padding: 20px;
                        }
                    """)
                    layout.addWidget(fallback_label)
                    self.field_widget = None
            else:
                # Fallback if mystical field not available
                fallback_label = QLabel("🔮 Field Unavailable\n\nMystical field module\nnot available")
                fallback_label.setAlignment(Qt.AlignCenter)
                fallback_label.setStyleSheet("""
                    QLabel {
                        color: #FF6347; 
                        font-size: 12px;
                        background-color: rgba(255, 99, 71, 0.1);
                        border: 1px solid #FF6347;
                        border-radius: 8px;
                        padding: 20px;
                    }
                """)
                layout.addWidget(fallback_label)
                self.field_widget = None
                
        except Exception as e:
            error_label = QLabel(f"🔮 Field Error\n\n{str(e)}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #FF6347; font-size: 12px;")
            layout.addWidget(error_label)
            self.field_widget = None
        
        return tab
    
    def create_inference_control_tab(self) -> QWidget:
        """Create the inference control tab for real model inference"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description
        desc_label = QLabel("🧠 Inference Mode - Load trained models and run real inference predictions")
        desc_label.setStyleSheet("""
            QLabel {
                color: #4169E1;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                background-color: rgba(65, 105, 225, 0.1);
                border: 1px solid #4169E1;
                border-radius: 5px;
                margin-bottom: 8px;
            }
        """)
        layout.addWidget(desc_label)
        
        # Model Selection Group
        model_group = QGroupBox("Model Selection")
        model_group.setStyleSheet("""
            QGroupBox {
                color: #87CEEB;
                font-weight: bold;
                border: 1px solid #4169E1;
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
        model_layout = QVBoxLayout(model_group)
        
        # Model dropdown
        model_row = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("color: #87CEEB; font-weight: bold; font-size: 11px;")
        model_label.setFixedWidth(100)
        model_row.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.setStyleSheet("""
            QComboBox {
                color: #87CEEB;
                background-color: #1E1E1E;
                border: 1px solid #4169E1;
                border-radius: 3px;
                padding: 4px;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                color: #87CEEB;
            }
        """)
        self.populate_model_list()
        model_row.addWidget(self.model_selector)
        
        # Refresh models button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4169E1;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A7FE1;
            }
        """)
        refresh_btn.clicked.connect(self.populate_model_list)
        model_row.addWidget(refresh_btn)
        
        model_layout.addLayout(model_row)
        
        # Model info display
        self.model_info_label = QLabel("Select a model to view details")
        self.model_info_label.setStyleSheet("""
            QLabel {
                color: #B0C4DE;
                font-size: 10px;
                padding: 4px;
                background-color: rgba(25, 25, 112, 0.2);
                border-radius: 3px;
                margin: 4px 0px;
            }
        """)
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)
        
        self.model_selector.currentTextChanged.connect(self.update_model_info)
        
        layout.addWidget(model_group)
        
        # Inference Controls Group
        controls_group = QGroupBox("Inference Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                color: #87CEEB;
                font-weight: bold;
                border: 1px solid #4169E1;
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
        
        # Load/Enable inference button
        self.inference_toggle_btn = QPushButton("Load Model & Start Inference")
        self.inference_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #228B22;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #32CD32;
            }
            QPushButton:disabled {
                background-color: #696969;
                color: #A9A9A9;
            }
        """)
        self.inference_toggle_btn.clicked.connect(self.toggle_real_inference)
        controls_layout.addWidget(self.inference_toggle_btn)
        
        # Inference rate control
        rate_row = QHBoxLayout()
        rate_label = QLabel("Inference Rate:")
        rate_label.setStyleSheet("color: #87CEEB; font-weight: bold; font-size: 11px;")
        rate_label.setFixedWidth(100)
        rate_row.addWidget(rate_label)
        
        self.inference_rate_slider = QSlider(Qt.Horizontal)
        self.inference_rate_slider.setRange(1, 10)  # 1-10 Hz for real models
        self.inference_rate_slider.setValue(2)  # Default 2 Hz for real inference
        self.inference_rate_slider.setFixedHeight(20)
        self.inference_rate_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4169E1;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #191970, stop:1 #4169E1);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00BFFF;
                border: 1px solid #4169E1;
                width: 14px;
                border-radius: 7px;
                margin: -4px 0;
            }
        """)
        rate_row.addWidget(self.inference_rate_slider)
        
        self.inference_rate_label = QLabel("2 Hz")
        self.inference_rate_label.setStyleSheet("color: #87CEEB; font-weight: bold; font-size: 11px;")
        self.inference_rate_label.setFixedWidth(40)
        rate_row.addWidget(self.inference_rate_label)
        
        self.inference_rate_slider.valueChanged.connect(self.update_inference_rate)
        
        controls_layout.addLayout(rate_row)
        layout.addWidget(controls_group)
        
        # Inference Status Group
        status_group = QGroupBox("Inference Status")
        status_group.setStyleSheet("""
            QGroupBox {
                color: #87CEEB;
                font-weight: bold;
                border: 1px solid #4169E1;
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
        status_layout = QVBoxLayout(status_group)
        
        # Status labels
        self.inference_status_label = QLabel("Model: Not loaded")
        self.inference_status_label.setStyleSheet("color: #FF6347; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.inference_status_label)
        
        self.prediction_count_label = QLabel("Predictions: 0")
        self.prediction_count_label.setStyleSheet("color: #87CEEB; font-size: 10px;")
        status_layout.addWidget(self.prediction_count_label)
        
        self.recursive_sessions_label = QLabel("Recorded Actions: 0")
        self.recursive_sessions_label.setStyleSheet("color: #87CEEB; font-size: 10px;")
        status_layout.addWidget(self.recursive_sessions_label)
        
        layout.addWidget(status_group)
        
        # Initialize state
        self.loaded_model = None
        self.model_metadata = None
        self.inference_active = False
        self.prediction_count = 0
        
        return tab
    
    def populate_model_list(self):
        """Populate the model selector with available trained models"""
        try:
            self.model_selector.clear()
            self.model_selector.addItem("-- Select a Model --")
            
            # Load model registry
            registry_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model_registry.json")
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                
                for model_name, metadata in registry.items():
                    # Format display name with key info
                    arch = metadata.get('variant_config', {}).get('architecture', 'unknown')
                    features = metadata.get('variant_config', {}).get('input_features', [])
                    loss = metadata.get('final_val_loss', 0)
                    display_name = f"{model_name} ({arch}) - Loss: {loss:.3f}"
                    self.model_selector.addItem(display_name, model_name)
                    
                self.log_message(f"🔄 Loaded {len(registry)} available models")
            else:
                self.log_message("⚠️ No model registry found")
                
        except Exception as e:
            self.log_message(f"⚠️ Error loading model list: {e}")
    
    def update_model_info(self, display_name):
        """Update model info display when selection changes"""
        try:
            current_data = self.model_selector.currentData()
            if not current_data:
                self.model_info_label.setText("Select a model to view details")
                return
                
            # Load model metadata
            registry_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model_registry.json")
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if current_data in registry:
                metadata = registry[current_data]
                config = metadata.get('variant_config', {})
                
                info_text = f"""Model: {current_data}
Architecture: {config.get('architecture', 'Unknown')}
Input Features: {', '.join(config.get('input_features', []))}
Hidden Size: {config.get('hidden_size', 'Unknown')}
Layers: {config.get('num_layers', 'Unknown')}
Training Loss: {metadata.get('final_loss', 'N/A'):.3f}
Validation Loss: {metadata.get('final_val_loss', 'N/A'):.3f}
Epochs Trained: {metadata.get('total_epochs', 'Unknown')}
Description: {config.get('description', 'No description available')}"""
                
                self.model_info_label.setText(info_text)
            else:
                self.model_info_label.setText("Model metadata not found")
                
        except Exception as e:
            self.model_info_label.setText(f"Error loading model info: {e}")
    
    def toggle_real_inference(self):
        """Toggle real model inference on/off"""
        if not self.inference_active:
            # Start inference
            selected_model = self.model_selector.currentData()
            if not selected_model:
                self.log_message("⚠️ Please select a model first")
                return
                
            self.load_and_start_inference(selected_model)
        else:
            # Stop inference
            self.stop_real_inference()
    
    def load_and_start_inference(self, model_name):
        """Load the selected model and start inference"""
        try:
            self.log_message(f"🔄 Loading model: {model_name}")
            self.inference_toggle_btn.setText("Loading...")
            self.inference_toggle_btn.setEnabled(False)
            
            # Load model registry for metadata
            registry_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "model_registry.json")
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if model_name not in registry:
                raise Exception(f"Model {model_name} not found in registry")
            
            self.model_metadata = registry[model_name]
            model_path = self.model_metadata['model_path']
            
            # Import model loading functionality
            from ml.model_manager import ModelManager
            from ml.advanced_inference import AdvancedInferenceEngine, MultiModelInferenceConfig
            
            # Load the model
            model_manager = ModelManager()
            self.loaded_model = model_manager.load_model_from_disk(model_path)
            
            # Create inference engine
            inference_config = MultiModelInferenceConfig(
                model_configs=[{
                    'model': self.loaded_model,
                    'metadata': self.model_metadata,
                    'weight': 1.0
                }],
                prediction_rate=self.inference_rate_slider.value(),
                sequence_length=self.model_metadata.get('variant_config', {}).get('sequence_length', 50),
                real_time=True
            )
            
            self.inference_engine = AdvancedInferenceEngine(inference_config)
            
            # Set up inference timer
            if not hasattr(self, 'inference_timer'):
                self.inference_timer = QTimer()
                self.inference_timer.timeout.connect(self.run_real_inference_cycle)
            
            # Start inference timer
            rate_hz = self.inference_rate_slider.value()
            interval_ms = int(1000 / rate_hz)
            self.inference_timer.start(interval_ms)
            
            # Update UI state
            self.inference_active = True
            self.prediction_count = 0
            self.inference_toggle_btn.setText("Stop Inference")
            self.inference_toggle_btn.setEnabled(True)
            self.inference_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #DC143C;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #FF6347;
                }
            """)
            
            self.inference_status_label.setText(f"Model: {model_name} (ACTIVE)")
            self.inference_status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 11px;")
            self.model_selector.setEnabled(False)
            
            self.log_message(f"✅ Model loaded and inference started at {rate_hz} Hz")
            
        except Exception as e:
            self.log_message(f"❌ Error loading model: {e}")
            self.inference_toggle_btn.setText("Load Model & Start Inference")
            self.inference_toggle_btn.setEnabled(True)
            self.inference_status_label.setText("Model: Load failed")
            self.inference_status_label.setStyleSheet("color: #FF6347; font-weight: bold; font-size: 11px;")
    
    def stop_real_inference(self):
        """Stop real model inference"""
        try:
            if hasattr(self, 'inference_timer'):
                self.inference_timer.stop()
            
            if hasattr(self, 'inference_engine'):
                del self.inference_engine
            
            if hasattr(self, 'loaded_model'):
                del self.loaded_model
            
            # Update UI state
            self.inference_active = False
            self.inference_toggle_btn.setText("Load Model & Start Inference")
            self.inference_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #228B22;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #32CD32;
                }
            """)
            
            self.inference_status_label.setText("Model: Not loaded")
            self.inference_status_label.setStyleSheet("color: #FF6347; font-weight: bold; font-size: 11px;")
            self.model_selector.setEnabled(True)
            
            self.log_message("🛑 Inference stopped and model unloaded")
            
        except Exception as e:
            self.log_message(f"⚠️ Error stopping inference: {e}")
    
    def run_real_inference_cycle(self):
        """Run a real inference cycle using the loaded model"""
        try:
            if not self.inference_active or not hasattr(self, 'inference_engine'):
                return
            
            # Collect recent data for inference
            recent_data = self.collect_recent_data_for_inference()
            
            if recent_data is None:
                # Not enough data yet
                return
            
            # Run inference
            prediction = self.inference_engine.predict(recent_data)
            
            if prediction is not None:
                self.prediction_count += 1
                self.prediction_count_label.setText(f"Predictions: {self.prediction_count}")
                
                # Process the real prediction
                self.process_real_prediction(prediction)
                
        except Exception as e:
            self.log_message(f"⚠️ Error in real inference cycle: {e}")
    
    def collect_recent_data_for_inference(self):
        """Collect recent RNG/EEG data for inference input"""
        try:
            sequence_length = self.model_metadata.get('variant_config', {}).get('sequence_length', 50)
            input_features = self.model_metadata.get('variant_config', {}).get('input_features', [])
            
            # This is a simplified version - in reality you'd collect from the data streams
            # For now, create synthetic data matching the expected input format
            input_data = {}
            
            if 'rng' in input_features:
                # Generate recent RNG-like data
                import numpy as np
                input_data['rng'] = np.random.rand(sequence_length, 8)  # 8 RNG channels
            
            if 'eeg' in input_features:
                # Generate recent EEG-like data  
                import numpy as np
                input_data['eeg'] = np.random.rand(sequence_length, 14)  # 14 EEG channels
            
            return input_data
            
        except Exception as e:
            self.log_message(f"⚠️ Error collecting inference data: {e}")
            return None
    
    def process_real_prediction(self, prediction):
        """Process real model predictions and draw them"""
        try:
            # Extract prediction components
            colors = prediction.get('colors', {})
            positions = prediction.get('positions', {})
            
            # Convert to drawing coordinates
            if 'r' in colors and 'g' in colors and 'b' in colors:
                r = max(0, min(255, int(colors['r'] * 255)))
                g = max(0, min(255, int(colors['g'] * 255)))
                b = max(0, min(255, int(colors['b'] * 255)))
            else:
                # Fallback colors
                r, g, b = 128, 128, 200
            
            # Get position from prediction or use center with small random offset
            if 'x' in positions and 'y' in positions:
                x = int(positions['x'] * self.canvas.width())
                y = int(positions['y'] * self.canvas.height())
            else:
                x = self.canvas.width() // 2 + random.randint(-100, 100)
                y = self.canvas.height() // 2 + random.randint(-100, 100)
            
            # Draw the real prediction
            self.draw_real_inference_prediction(x, y, (r, g, b, 180))
            
            # Record the action if session is active
            if (hasattr(self, 'data_logger') and self.data_logger and 
                self.session_active):
                
                inference_action = DrawingAction(
                    timestamp=time.time(),
                    action_type='real_inference_prediction',
                    position=(x, y),
                    color=(r, g, b, 180),
                    brush_size=10,
                    pressure=0.8,
                    consciousness_layer=self.current_layer,
                    pocket_dimension=self.pocket_dimension,
                    metadata={
                        'model_name': self.model_selector.currentData(),
                        'prediction_data': prediction,
                        'inference_mode': 'real_model',
                        'prediction_count': self.prediction_count
                    }
                )
                
                self.data_logger.log_drawing_action(inference_action)
                self.recursive_sessions_label.setText(f"Recorded Actions: {self.prediction_count}")
            
            self.log_message(f"🧠 Real prediction: ({x},{y}) color=({r},{g},{b}) from {self.model_selector.currentData()}")
            
        except Exception as e:
            self.log_message(f"⚠️ Error processing real prediction: {e}")
    
    def draw_real_inference_prediction(self, x, y, color):
        """Draw real inference prediction on canvas"""
        try:
            painter = QPainter(self.canvas.layers[self.current_layer])
            
            r, g, b, alpha = color
            prediction_color = QColor(r, g, b, alpha)
            
            # Draw main prediction mark (diamond shape for real predictions)
            points = [
                QPoint(x, y-8),      # top
                QPoint(x+8, y),      # right  
                QPoint(x, y+8),      # bottom
                QPoint(x-8, y)       # left
            ]
            
            brush = QBrush(prediction_color)
            painter.setBrush(brush)
            painter.setPen(QPen(prediction_color, 2))
            painter.drawPolygon(points)
            
            # Draw outer glow
            glow_color = QColor(r, g, b, alpha//4)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(QPen(glow_color, 1))
            painter.drawEllipse(x-15, y-15, 30, 30)
            
            painter.end()
            
            # Update canvas display
            self.canvas.update()
            
        except Exception as e:
            self.log_message(f"⚠️ Error drawing real prediction: {e}")
        
        self.inference_rate_slider.valueChanged.connect(self.update_inference_rate)
        
        controls_layout.addLayout(rate_row)
        layout.addWidget(controls_group)
        
        # Inference status
        status_group = QGroupBox("Inference Status")
        status_group.setStyleSheet("""
            QGroupBox {
                color: #87CEEB;
                font-weight: bold;
                border: 1px solid #4169E1;
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
        status_layout = QVBoxLayout(status_group)
        
        self.inference_status_label = QLabel("Inference: Disabled")
        self.inference_status_label.setStyleSheet("color: #FF6347; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.inference_status_label)
        
        self.recursive_sessions_label = QLabel("Recursive Sessions: 0")
        self.recursive_sessions_label.setStyleSheet("color: #87CEEB; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.recursive_sessions_label)
        
        layout.addWidget(status_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return tab
        
    def setup_timers(self):
        """Set up update timers"""
        # Session time update
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_session_time)
        
        # Hardware data update
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_hardware_data)
        self.data_timer.start(50)  # 20 Hz update rate
        
        # Mystical field update timer (ensures continuous field activity)
        self.field_timer = QTimer()
        self.field_timer.timeout.connect(self.update_mystical_field)
        self.field_timer.start(100)  # 10 Hz update rate for field
    
    def update_mystical_field(self):
        """Ensure mystical field stays active with continuous updates"""
        if hasattr(self, 'field_widget') and self.field_widget:
            # Update particle count display
            self.update_particle_count()
            
            # Generate synthetic field data if no RNG is active
            if not (self.rng_device and self.rng_device.is_connected and self.rng_device.is_streaming):
                self.generate_synthetic_field_data()
        
    def set_hardware_devices(self, rng_device, eeg_bridge):
        """Set hardware device references"""
        self.rng_device = rng_device
        self.eeg_bridge = eeg_bridge  # Now using EEG bridge instead of direct device
        self.update_hardware_status()
        
    def start_session(self):
        """Start data collection session"""
        self.session_active = True
        self.session_start_time = time.time()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Start data logger session if available
        if hasattr(self, 'data_logger') and self.data_logger:
            try:
                # Enhanced session metadata for merged mode
                experiment_notes = "Mystical consciousness data generation"
                if hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled:
                    experiment_notes += " with recursive inference"
                
                session_id = self.data_logger.start_session(
                    participant_id="consciousness_session",
                    experiment_notes=experiment_notes,
                    hardware_config={
                        'rng_device': 'TrueRNG_V3' if self.rng_device else 'None',
                        'eeg_device': 'EEG_Bridge' if self.eeg_bridge else 'None',
                        'mystical_field': 'Enabled' if hasattr(self, 'field_widget') and self.field_widget else 'Disabled',
                        'inference_mode': 'Enabled' if hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled else 'Disabled',
                        'recursive_recording': 'Enabled' if hasattr(self, 'recursive_recording_checkbox') and self.recursive_recording_checkbox.isChecked() else 'Disabled'
                    }
                )
                self.log_message(f"📊 Data logging started: {session_id}")
                
                # Log merged mode configuration
                if hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled:
                    self.log_message("🧠 Merged mode: Recording with recursive inference")
                if hasattr(self, 'field_widget') and self.field_widget:
                    self.log_message("🔮 Mystical field visualization active")
                    
            except Exception as e:
                self.log_message(f"Error starting data logger: {e}")
        
        # Start hardware streaming
        if self.rng_device and self.rng_device.is_connected:
            self.rng_device.start_streaming()
            
        # EEG bridge is already streaming when connected
        if self.eeg_bridge and self.eeg_bridge.is_streaming():
            self.log_message("EEG already streaming")
            
        self.session_timer.start(1000)  # Update every second
        
        self.log_message("Session started")
        
    def stop_session(self):
        """Stop data collection session"""
        self.session_active = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Stop data logger session if available
        if hasattr(self, 'data_logger') and self.data_logger:
            try:
                output_path = self.data_logger.stop_session()
                if output_path:
                    self.log_message(f"📁 Data saved to: {output_path}")
                    
                    # Log merged mode statistics
                    if hasattr(self, 'recursive_session_count') and self.recursive_session_count > 0:
                        self.log_message(f"🔄 Recursive inference sessions recorded: {self.recursive_session_count}")
                        self.log_message("🌀 Consciousness feedback loops captured for analysis")
                        
                    if hasattr(self, 'field_widget') and self.field_widget:
                        particle_count = len(self.field_widget.field.particles) if hasattr(self.field_widget, 'field') else 0
                        self.log_message(f"🔮 Mystical field particles: {particle_count}")
                        
                else:
                    self.log_message("No active data logging session to stop")
            except Exception as e:
                self.log_message(f"Error stopping data logger: {e}")
        
        # Stop inference components if running
        if hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled:
            self.stop_inference_components()
            
        # Reset recursive session count for next session
        if hasattr(self, 'recursive_session_count'):
            self.recursive_session_count = 0
            if hasattr(self, 'recursive_sessions_label'):
                self.recursive_sessions_label.setText("Recursive Sessions: 0")
        
        # Stop hardware streaming
        if self.rng_device:
            self.rng_device.stop_streaming()
            
        # EEG bridge continues streaming (managed by bridge)
        # No need to explicitly stop unless shutting down
            
        self.session_timer.stop()
        
        self.log_message("Session stopped")
        
    def choose_color(self):
        """Open color chooser dialog"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.set_color(color)
            self.color_button.setStyleSheet(f"background-color: {color.name()}")
            
    def set_quick_color(self, hex_color: str):
        """Set color from quick palette"""
        color = QColor(hex_color)
        self.canvas.set_color(color)
        self.color_button.setStyleSheet(f"background-color: {hex_color}")
        
        # Update button highlighting
        for value in self.color_buttons.values():
            # Handle both button objects and (hex_color, btn) tuples
            btn = value[1] if isinstance(value, tuple) else value
            if hasattr(btn, 'setStyleSheet'):
                btn.setStyleSheet(btn.styleSheet().replace("border: 3px solid #000;", "border: 2px solid #333;"))
        
        # Highlight selected color button
        for name, value in self.color_buttons.items():
            # Handle both button objects and (hex_color, btn) tuples
            btn = value[1] if isinstance(value, tuple) else value
            if hasattr(btn, 'styleSheet') and hex_color.upper() in btn.styleSheet().upper():
                btn.setStyleSheet(btn.styleSheet().replace("border: 2px solid #333;", "border: 3px solid #000;"))
                break
            
    def brush_size_changed(self, value):
        """Handle brush size slider change"""
        self.canvas.set_brush_size(value)
        self.brush_size_label.setText(str(value))
        
        # Update button highlighting
        for size, btn in self.brush_size_buttons.items():
            if size == value:
                btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            else:
                btn.setStyleSheet("")
                
    def set_quick_brush_size(self, size: int):
        """Set brush size from quick buttons"""
        self.canvas.set_brush_size(size)
        self.brush_slider.setValue(size)
        self.brush_size_label.setText(str(size))
        
        # Update button highlighting
        for s, btn in self.brush_size_buttons.items():
            if s == size:
                btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            else:
                btn.setStyleSheet("")
    
    def toggle_dial_visualization(self, state):
        """Toggle the interlocking dial visualization overlay"""
        enabled = (state == Qt.Checked)
        self.canvas.set_dial_visualization(enabled)
        
        if enabled:
            logging.info("✨ Interlocking dial visualization enabled - white curves will show 3D geometry")
        else:
            logging.info("Interlocking dial visualization disabled")
                
    def set_consciousness_layer(self, layer: int):
        """Set the active consciousness layer (1, 2, or 3)"""
        self.current_layer = layer
        self.canvas.set_consciousness_layer(layer)
        
        # Mystical layer color scheme
        layer_info = {
            1: {"color": "#9D4EDD", "name": "Ethereal Consciousness"},
            2: {"color": "#3D5A80", "name": "Cosmic Flow"}, 
            3: {"color": "#4B0082", "name": "Shadow Realm"}
        }
        
        # Update button highlighting with mystical glow
        for l, btn in self.layer_buttons.items():
            info = layer_info[l]
            if l == layer:
                btn.setChecked(True)
                btn.setStyleSheet(f"""
                    background-color: {info['color']}; 
                    color: white; 
                    font-weight: bold;
                    border: 2px solid #1a1a1a;
                    border-radius: 3px;
                    box-shadow: 0 0 8px {info['color']}60;
                """)
            else:
                btn.setChecked(False)
                btn.setStyleSheet(f"""
                    background-color: #2a2a2a; 
                    color: {info['color']}; 
                    border: 1px solid {info['color']};
                    border-radius: 3px;
                """)
        
        # Update info label with mystical styling
        current_info = layer_info[layer]
        self.layer_info_label.setText(f"Current: Layer {layer} ({current_info['name']})")
        self.layer_info_label.setStyleSheet(f"""
            font-weight: bold; 
            color: {current_info['color']}; 
            background-color: #1a1a1a; 
            padding: 5px; 
            border-radius: 3px;
            border: 1px solid {current_info['color']};
        """)
        
    def handle_pocket_dimension_navigation(self, button_pressed: int, use_layer: int = None):
        """Handle pocket dimension navigation based on specified layer and button pressed"""
        # Use specified layer or current layer
        navigation_layer = use_layer if use_layer is not None else self.current_layer
        old_dimension = self.pocket_dimension
        
        # Pocket dimension navigation rules (your original specification):
        # Layer 1: Press 2 (+1), Press 3 (+2)
        # Layer 2: Press 1 (+1), Press 3 (-1)  
        # Layer 3: Press 1 (-1), Press 2 (-2)
        
        change = 0
        if navigation_layer == 1:
            if button_pressed == 2:
                change = +1
            elif button_pressed == 3:
                change = +2
        elif navigation_layer == 2:
            if button_pressed == 1:
                change = +1
            elif button_pressed == 3:
                change = -1
        elif navigation_layer == 3:
            if button_pressed == 1:
                change = -1
            elif button_pressed == 2:
                change = -2
        
        # Apply change
        self.pocket_dimension += change
        
        # Debug output to confirm value update
        print(f"DEBUG: Pocket dimension updated: {old_dimension} + {change} = {self.pocket_dimension}")
        
        # Update main display - always show just the current dimension clearly
        self.pocket_display.setText(f"Dimension: {self.pocket_dimension}")
        
        # Update static net change display - this stays visible and shows what just happened
        if change != 0:
            change_sign = "+" if change > 0 else ""
            self.net_change_display.setText(f"Net: {change_sign}{change}")
            # Make it flash briefly to draw attention
            self.flash_net_change_display()
        else:
            self.net_change_display.setText("Net: --")
        
        # Update info label with very clear change notification
        if change != 0:
            change_text = f"({change:+d})"
            if change == 1:
                change_desc = "PLUS ONE"
            elif change == -1:
                change_desc = "MINUS ONE"
            elif change == 2:
                change_desc = "PLUS TWO"
            elif change == -2:
                change_desc = "MINUS TWO"
            else:
                change_desc = f"{'PLUS' if change > 0 else 'MINUS'} {abs(change)}"
            
            self.pocket_info_label.setText(f"🌀 {change_desc} → Dimension {old_dimension} → {self.pocket_dimension}")
        else:
            self.pocket_info_label.setText(f"Layer {navigation_layer} + Button {button_pressed} = no change (invalid combination)")
        
        # Log the dimensional shift for consciousness tracking
        print(f"🌀 DIMENSIONAL SHIFT: {old_dimension} → {self.pocket_dimension} (Layer {navigation_layer}, Button {button_pressed}, Change: {change})")
        
        # Show visual feedback
        self.show_dimension_change_feedback()
        
        # Add dimensional data to session if we have a data logger
        if hasattr(self, 'data_logger') and self.data_logger:
            # Create a drawing action to represent the dimensional change
            dimensional_action = DrawingAction(
                timestamp=time.time(),
                action_type='dimension_change',
                position=(0, 0),  # No position for dimension changes
                color=(255, 215, 0, 255),  # Gold color for dimension events
                brush_size=1,
                pressure=1.0,
                consciousness_layer=navigation_layer,
                pocket_dimension=self.pocket_dimension,
                metadata={
                    'old_dimension': old_dimension,
                    'new_dimension': self.pocket_dimension,
                    'button_pressed': button_pressed,
                    'change': change
                }
            )
            self.data_logger.log_drawing_action(dimensional_action)
        
    def handle_layer_and_dimension_change(self, target_layer: int, force_layer_change: bool = False):
        """Handle both layer change and pocket dimension navigation"""
        current_layer = self.current_layer
        
        # Debug output
        print(f"DEBUG: Current layer: {current_layer}, Target layer: {target_layer}, Force layer change: {force_layer_change}")
        
        # ALWAYS change to the target layer first
        print(f"DEBUG: Changing layer from {current_layer} to {target_layer}")
        self.set_consciousness_layer(target_layer)
        
        # THEN check if we should also navigate dimensions based on the OLD layer's rules
        # (using the layer we were on before switching)
        should_navigate_dimension = False
        if current_layer == 1 and target_layer in [2, 3]:  # Was on Layer 1: buttons 2,3 navigate
            should_navigate_dimension = True
        elif current_layer == 2 and target_layer in [1, 3]:  # Was on Layer 2: buttons 1,3 navigate  
            should_navigate_dimension = True
        elif current_layer == 3 and target_layer in [1, 2]:  # Was on Layer 3: buttons 1,2 navigate
            should_navigate_dimension = True
        
        print(f"DEBUG: Should navigate dimension: {should_navigate_dimension}")
        
        if should_navigate_dimension:
            # Navigate dimensions using the OLD layer's rules
            print(f"DEBUG: ALSO calling dimension navigation with button {target_layer} (from old layer {current_layer})")
            self.handle_pocket_dimension_navigation(target_layer, use_layer=current_layer)
        
        # Update info to show current layer navigation options
        if target_layer == 1:
            self.pocket_info_label.setText("Layer 1: Click 2(+1) or 3(+2) to switch layer AND navigate dimensions")
        elif target_layer == 2:
            self.pocket_info_label.setText("Layer 2: Click 1(+1) or 3(-1) to switch layer AND navigate dimensions")
        elif target_layer == 3:
            self.pocket_info_label.setText("Layer 3: Click 1(-1) or 2(-2) to switch layer AND navigate dimensions")
            
    def show_dimension_change_feedback(self):
        """Show visual feedback for pocket dimension changes"""
        # Flash the pocket display to show change occurred
        original_style = self.pocket_display.styleSheet()
        flash_style = """
            font-weight: bold; 
            color: #000000; 
            background-color: #00FF00; 
            padding: 10px; 
            border: 4px solid #FFD700; 
            border-radius: 8px;
            text-align: center;
            font-size: 14px;
        """
        
        # Flash effect with longer duration
        self.pocket_display.setStyleSheet(flash_style)
        
        # Use QTimer to restore original style after 800ms
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(800, lambda: self.pocket_display.setStyleSheet(original_style))
        
        # Also flash the info label with bright color
        original_info_style = self.pocket_info_label.styleSheet()
        flash_info_style = "font-size: 12px; color: #00FF00; font-style: italic; font-weight: bold; background-color: #1a1a1a; padding: 3px;"
        
        self.pocket_info_label.setStyleSheet(flash_info_style)
        QTimer.singleShot(1000, lambda: self.pocket_info_label.setStyleSheet(original_info_style))
        
    def flash_net_change_display(self):
        """Flash the net change display to draw attention to the change"""
        original_style = self.net_change_display.styleSheet()
        flash_style = """
            font-weight: bold; 
            color: #000000; 
            background-color: #00FF7F; 
            padding: 6px; 
            border: 3px solid #FFFFFF; 
            border-radius: 5px;
            text-align: center;
            font-size: 12px;
        """
        
        # Apply flash style
        self.net_change_display.setStyleSheet(flash_style)
        
        # Timer to revert to original style
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, lambda: self.net_change_display.setStyleSheet(original_style))
        
    def opacity_changed(self, value):
        """Handle opacity slider change"""
        logging.debug(f"Opacity slider changed to {value}")
        self.canvas.set_brush_opacity(value)
        percentage = int((value / 255.0) * 100)
        self.opacity_label.setText(f"{percentage}%")
        logging.debug(f"Opacity set to {percentage}% (value: {value})")
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear_canvas()
        self.log_message("Canvas cleared")
        
    def on_drawing_action(self, action: DrawingAction):
        """Handle drawing action from canvas"""
        if not self.session_active:
            return
            
        # Inject current hardware data
        if self.rng_device:
            latest_rng = self.rng_device.get_latest_samples(1)
            if latest_rng:
                action.rng_data = latest_rng[0].normalized
                
        if self.eeg_device:
            latest_eeg = self.eeg_device.get_latest_samples(1)
            if latest_eeg:
                action.eeg_data = latest_eeg[0].channels
                
        # Log action to data logger if available
        if hasattr(self, 'data_logger') and self.data_logger:
            try:
                self.data_logger.log_drawing_action(action)
            except Exception as e:
                self.log_message(f"Error logging drawing action: {e}")
                
        # Log action (could save to file here)
        if action.action_type == 'stroke_start':
            self.log_message(f"Started drawing at ({action.position[0]}, {action.position[1]})")
            
    def update_session_time(self):
        """Update session time display"""
        if self.session_active:
            elapsed = time.time() - self.session_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.session_time_label.setText(f"Session Time: {minutes:02d}:{seconds:02d}")
            
    def update_hardware_data(self):
        """Update hardware data displays"""
        if not self.session_active:
            return
            
        # Update RNG data visualization
        if self.rng_device:
            latest_rng = self.rng_device.get_latest_samples(10)
            if latest_rng:
                for sample in latest_rng:
                    self.data_viz.add_rng_data(sample.normalized[:4])  # First 4 values
                    
                    # Log to data logger if available
                    if hasattr(self, 'data_logger') and self.data_logger:
                        try:
                            self.data_logger.log_rng_sample(sample)
                        except Exception as e:
                            # Only log error once to avoid spam
                            if not hasattr(self, '_rng_log_error_shown'):
                                self.log_message(f"Error logging RNG data: {e}")
                                self._rng_log_error_shown = True
                    
        # Update EEG data visualization
        if self.eeg_device:
            latest_eeg = self.eeg_device.get_latest_samples(5)  # Get fewer samples for better performance
            if latest_eeg:
                for sample in latest_eeg:
                    # Pass the entire channel dictionary
                    self.data_viz.add_eeg_data(sample.channels)
                    
                    # Log to data logger if available
                    if hasattr(self, 'data_logger') and self.data_logger:
                        try:
                            self.data_logger.log_eeg_sample(sample)
                        except Exception as e:
                            # Only log error once to avoid spam
                            if not hasattr(self, '_eeg_log_error_shown'):
                                self.log_message(f"Error logging EEG data: {e}")
                                self._eeg_log_error_shown = True
                    
        # Update counters
        self.drawing_actions_label.setText(f"Drawing Actions: {len(self.canvas.drawing_actions)}")
        
        if self.rng_device:
            rng_count = self.rng_device.data_queue.qsize()
            self.rng_samples_label.setText(f"RNG Samples: {rng_count}")
            
        if self.eeg_device:
            eeg_count = self.eeg_device.data_queue.qsize()
            self.eeg_samples_label.setText(f"EEG Samples: {eeg_count}")
            
    def update_hardware_status(self):
        """Update hardware connection status"""
        # Check RNG status
        if self.rng_device and hasattr(self.rng_device, 'is_connected'):
            try:
                if self.rng_device.is_connected:
                    self.rng_status.setText("RNG: Connected")
                    self.rng_status.setStyleSheet("color: green")
                else:
                    self.rng_status.setText("RNG: Disconnected")
                    self.rng_status.setStyleSheet("color: red")
            except Exception as e:
                self.rng_status.setText("RNG: Error")
                self.rng_status.setStyleSheet("color: red")
        else:
            self.rng_status.setText("RNG: Disconnected")
            self.rng_status.setStyleSheet("color: red")
            
        # Check EEG status using EEG bridge
        if self.eeg_bridge is None:
            self.eeg_status.setText("EEG: Disabled")
            self.eeg_status.setStyleSheet("color: gray")
        else:
            try:
                connection_info = self.eeg_bridge.get_connection_info()
                source_name = connection_info.source.value.upper()
                
                if connection_info.status.value == "streaming":
                    signal_quality = connection_info.signal_quality
                    quality_text = f" (Q: {signal_quality:.1f})" if signal_quality > 0 else ""
                    self.eeg_status.setText(f"EEG: {source_name} Streaming{quality_text}")
                    self.eeg_status.setStyleSheet("color: lime")
                elif connection_info.status.value == "connected":
                    self.eeg_status.setText(f"EEG: {source_name} Connected")
                    self.eeg_status.setStyleSheet("color: green")
                elif connection_info.status.value == "connecting":
                    self.eeg_status.setText(f"EEG: {source_name} Connecting...")
                    self.eeg_status.setStyleSheet("color: orange")
                elif connection_info.status.value == "error":
                    error_msg = connection_info.error_message or "Unknown error"
                    self.eeg_status.setText(f"EEG: {source_name} Error")
                    self.eeg_status.setStyleSheet("color: red")
                    self.eeg_status.setToolTip(error_msg)
                else:
                    self.eeg_status.setText(f"EEG: {source_name} Disconnected")
                    self.eeg_status.setStyleSheet("color: red")
                    
            except Exception as e:
                self.eeg_status.setText("EEG: Error")
                self.eeg_status.setStyleSheet("color: red")
                self.logger.warning(f"Error checking EEG status: {e}")
    
    # Mystical Field Methods for Merged Mode
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
            
            self.log_message(f"Mystical field intensity: {intensity:.2f}")
    
    def reset_mystical_field(self):
        """Reset the mystical field to initial state"""
        if hasattr(self, 'field_widget') and self.field_widget:
            # Clear existing particles and recreate initial ones
            self.field_widget.field.particles.clear()
            self.field_widget.field._create_initial_particles()
            self.log_message("🌟 Mystical field reset")
    
    def update_particle_count(self):
        """Update particle count display"""
        if hasattr(self, 'field_widget') and self.field_widget and hasattr(self, 'particle_count_label'):
            count = len(self.field_widget.field.particles)
            self.particle_count_label.setText(f"Particles: {count}")
    
    def feed_rng_to_mystical_field(self, rng_bytes: bytes):
        """Feed RNG data to the mystical field for visualization"""
        if hasattr(self, 'field_widget') and self.field_widget:
            self.field_widget.add_rng_data(rng_bytes)
            self.update_particle_count()
    
    # Inference Mode Methods for Recursive Recording
    def toggle_inference_mode(self, enabled: bool):
        """Toggle inference mode on/off"""
        self.inference_mode_enabled = enabled
        
        if enabled:
            self.inference_status_label.setText("Inference: Enabled")
            self.inference_status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 11px;")
            self.log_message("🧠 Inference mode enabled - recursive recording active")
            
            # Initialize inference components if needed
            self.setup_inference_components()
            
        else:
            self.inference_status_label.setText("Inference: Disabled")
            self.inference_status_label.setStyleSheet("color: #FF6347; font-weight: bold; font-size: 11px;")
            self.log_message("🛑 Inference mode disabled")
            
            # Stop inference components if running
            self.stop_inference_components()
    
    def setup_inference_components(self):
        """Initialize inference components for merged mode"""
        try:
            # For now, create a simple mock inference system
            # This avoids complex model loading issues in merged mode
            self.inference_engine = MockInferenceEngine()
            
            # Set up inference timer for regular predictions
            if not hasattr(self, 'inference_timer'):
                self.inference_timer = QTimer()
                self.inference_timer.timeout.connect(self.run_inference_cycle)
            
            # Get inference rate from slider (default 5 Hz)
            inference_rate = getattr(self, 'inference_rate_slider', None)
            if inference_rate:
                rate_hz = inference_rate.value()
            else:
                rate_hz = 5  # Default 5 Hz
            
            # Start inference timer
            interval_ms = int(1000 / rate_hz)  # Convert Hz to milliseconds
            self.inference_timer.start(interval_ms)
            
            self.log_message(f"🔧 Mock inference engine initialized (rate: {rate_hz} Hz)")
                
        except Exception as e:
            self.log_message(f"⚠️ Error setting up inference: {e}")
            self.enable_inference_checkbox.setChecked(False)
            self.toggle_inference_mode(False)
    
    def stop_inference_components(self):
        """Stop inference components"""
        if hasattr(self, 'inference_timer'):
            self.inference_timer.stop()
            self.log_message("⏹️ Inference timer stopped")
            
        if hasattr(self, 'inference_engine'):
            try:
                self.inference_engine.stop()
                self.log_message("🛑 Inference engine stopped")
            except Exception as e:
                self.log_message(f"⚠️ Error stopping inference: {e}")
    
    def update_inference_rate(self, rate_hz):
        """Update the inference rate when slider changes"""
        try:
            # Update the rate label
            if hasattr(self, 'inference_rate_label'):
                self.inference_rate_label.setText(f"{rate_hz} Hz")
            
            # Update timer interval if inference is active
            if (hasattr(self, 'inference_timer') and 
                hasattr(self, 'inference_mode_enabled') and 
                self.inference_mode_enabled):
                
                interval_ms = int(1000 / rate_hz)
                self.inference_timer.setInterval(interval_ms)
                self.log_message(f"🔧 Inference rate updated to {rate_hz} Hz")
                
        except Exception as e:
            self.log_message(f"⚠️ Error updating inference rate: {e}")
    
    def run_inference_cycle(self):
        """Run a single inference cycle for merged mode"""
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                # Generate mock prediction data for testing
                prediction_data = {
                    'colors': {
                        'r': 128 + int(50 * math.sin(time.time() * 0.5)),
                        'g': 128 + int(50 * math.cos(time.time() * 0.7)),
                        'b': 128 + int(50 * math.sin(time.time() * 0.3)),
                        'alpha': 255
                    },
                    'dials': {
                        'dial_1': {'value': (math.sin(time.time() * 0.4) + 1) / 2},
                        'dial_2': {'value': (math.cos(time.time() * 0.6) + 1) / 2}
                    }
                }
                
                self.on_inference_prediction(prediction_data)
                
        except Exception as e:
            self.log_message(f"⚠️ Error in inference cycle: {e}")
    
    def on_inference_prediction(self, prediction_data):
        """Handle inference predictions for recursive recording"""
        try:
            # VISUAL RENDERING: Draw inference predictions on canvas
            self.draw_inference_prediction(prediction_data)
            
            # If recursive recording is enabled, record this inference session
            if (self.recursive_recording_checkbox.isChecked() and 
                hasattr(self, 'data_logger') and self.data_logger and
                self.session_active):
                
                # Extract color from prediction for visual action
                colors = prediction_data.get('colors', {})
                inference_color = (
                    colors.get('r', 128),
                    colors.get('g', 128), 
                    colors.get('b', 128),
                    colors.get('alpha', 200)  # Semi-transparent for inference
                )
                
                # Generate position based on dials for spatial inference
                dials = prediction_data.get('dials', {})
                dial_1 = dials.get('dial_1', {}).get('value', 0.5)
                dial_2 = dials.get('dial_2', {}).get('value', 0.5)
                inference_position = (
                    int(dial_1 * self.canvas.width()),
                    int(dial_2 * self.canvas.height())
                )
                
                # Create inference action for recursive recording
                inference_action = DrawingAction(
                    timestamp=time.time(),
                    action_type='inference_prediction',
                    position=inference_position,
                    color=inference_color,
                    brush_size=8,  # Larger brush for inference marks
                    pressure=0.7,
                    consciousness_layer=self.current_layer,
                    pocket_dimension=self.pocket_dimension,
                    metadata={
                        'prediction_type': 'recursive_inference',
                        'prediction_data': prediction_data,
                        'inference_mode': 'merged_generation',
                        'recursive_session_count': getattr(self, 'recursive_session_count', 0)
                    }
                )
                
                # Log the inference action
                self.data_logger.log_drawing_action(inference_action)
                
                # Increment recursive session count
                if not hasattr(self, 'recursive_session_count'):
                    self.recursive_session_count = 0
                self.recursive_session_count += 1
                self.recursive_sessions_label.setText(f"Recursive Sessions: {self.recursive_session_count}")
                
                # Feed inference data back into the system for recursive effect
                self.process_recursive_inference_feedback(prediction_data)
                
        except Exception as e:
            self.log_message(f"⚠️ Error processing inference prediction: {e}")
    
    def draw_inference_prediction(self, prediction_data):
        """Draw inference predictions visually on the canvas"""
        try:
            # Extract color prediction
            colors = prediction_data.get('colors', {})
            r = colors.get('r', 128)
            g = colors.get('g', 128)
            b = colors.get('b', 128)
            alpha = colors.get('alpha', 180)  # Semi-transparent
            
            # Extract position from dials
            dials = prediction_data.get('dials', {})
            dial_1 = dials.get('dial_1', {}).get('value', 0.5)
            dial_2 = dials.get('dial_2', {}).get('value', 0.5)
            
            # Convert dials to canvas coordinates
            x = int(dial_1 * self.canvas.width())
            y = int(dial_2 * self.canvas.height())
            
            # Draw inference prediction as a glowing circle
            painter = QPainter(self.canvas.layers[self.current_layer])
            
            # Set up inference brush with glow effect
            inference_color = QColor(r, g, b, alpha)
            
            # Draw main inference mark
            brush = QBrush(inference_color)
            painter.setBrush(brush)
            painter.setPen(QPen(inference_color, 2))
            painter.drawEllipse(x-6, y-6, 12, 12)
            
            # Draw outer glow
            glow_color = QColor(r, g, b, alpha//3)
            painter.setBrush(QBrush(glow_color))
            painter.setPen(QPen(glow_color, 1))
            painter.drawEllipse(x-12, y-12, 24, 24)
            
            painter.end()
            
            # Update canvas display
            self.canvas.update()
            
            # Log visual inference
            self.log_message(f"🧠 Inference drawn: ({x},{y}) color=({r},{g},{b})")
            
        except Exception as e:
            self.log_message(f"⚠️ Error drawing inference prediction: {e}")
    
    def process_recursive_inference_feedback(self, prediction_data):
        """Process inference feedback for recursive consciousness simulation"""
        try:
            # Use inference predictions to influence the mystical field
            if hasattr(self, 'field_widget') and self.field_widget:
                # Convert prediction data to influence field parameters
                if 'colors' in prediction_data:
                    colors = prediction_data['colors']
                    # Use color predictions to create synthetic RNG data
                    synthetic_rng = bytes([
                        int(colors.get('r', 128)),
                        int(colors.get('g', 128)), 
                        int(colors.get('b', 128)),
                        int(colors.get('alpha', 255)),
                        int((colors.get('r', 128) + colors.get('g', 128)) % 256),
                        int((colors.get('g', 128) + colors.get('b', 128)) % 256),
                        int((colors.get('b', 128) + colors.get('r', 128)) % 256),
                        int((colors.get('alpha', 255) + colors.get('r', 128)) % 256)
                    ])
                    
                    # Feed synthetic data to mystical field
                    self.feed_rng_to_mystical_field(synthetic_rng)
            
            # Influence pocket dimension navigation based on inference
            if 'dials' in prediction_data:
                dials = prediction_data['dials']
                # Use dial predictions to occasionally trigger dimension shifts
                if any(abs(dial.get('value', 0.5) - 0.5) > 0.3 for dial in dials.values()):
                    # Strong dial deviation triggers dimension shift
                    if not hasattr(self, 'last_inference_dimension_shift'):
                        self.last_inference_dimension_shift = 0
                    
                    current_time = time.time()
                    if current_time - self.last_inference_dimension_shift > 5.0:  # Max once per 5 seconds
                        # Trigger dimension navigation based on strongest dial
                        strongest_dial = max(dials.keys(), 
                                           key=lambda k: abs(dials[k].get('value', 0.5) - 0.5))
                        button_to_press = int(strongest_dial.split('_')[-1]) if '_' in strongest_dial else 2
                        
                        self.handle_pocket_dimension_navigation(button_to_press)
                        self.log_message(f"🌀 Inference-triggered dimension shift via button {button_to_press}")
                        self.last_inference_dimension_shift = current_time
                        
        except Exception as e:
            self.log_message(f"⚠️ Error processing recursive feedback: {e}")
            
    # Enhanced hardware data processing for merged mode
    def update_hardware_data(self):
        """Update hardware data displays and feed to mystical field"""
        # RNG processing - feed to mystical field even without active session
        if self.rng_device and self.rng_device.is_connected:
            try:
                # Get RNG samples and log them
                samples = self.rng_device.read_all_samples()
                for sample in samples:
                    # Log only if session is active
                    if self.session_active and hasattr(self, 'data_logger') and self.data_logger:
                        self.data_logger.log_rng_sample(sample)
                        
                    # Always feed RNG data to mystical field for visualization
                    if len(sample.raw_bytes) > 0:
                        self.feed_rng_to_mystical_field(sample.raw_bytes)
                        
                    # Also feed to data visualization
                    if hasattr(self, 'data_viz'):
                        self.data_viz.add_rng_data(sample.normalized)
                        
            except Exception as e:
                self.logger.warning(f"Error reading RNG samples: {e}")
        
        # If no RNG data available, generate synthetic data to keep mystical field active
        elif hasattr(self, 'field_widget') and self.field_widget:
            self.generate_synthetic_field_data()
                
        # Enhanced EEG processing (original code preserved)
        if self.eeg_bridge and self.eeg_bridge.is_streaming():
            try:
                # Get EEG samples and log them
                samples = self.eeg_bridge.read_all_samples()
                for sample in samples:
                    # Log only if session is active
                    if self.session_active and hasattr(self, 'data_logger') and self.data_logger:
                        self.data_logger.log_eeg_sample(sample)
                        
                    # Feed to data visualization
                    if hasattr(self, 'data_viz'):
                        # Convert EEG sample to expected format
                        eeg_dict = {}
                        if hasattr(sample, 'channels') and sample.channels:
                            for channel, value in sample.channels.items():
                                eeg_dict[channel] = value
                        self.data_viz.add_eeg_data(eeg_dict)
                        
            except Exception as e:
                self.logger.warning(f"Error reading EEG samples: {e}")
                
        # Update data visualization
        if hasattr(self, 'data_viz'):
            self.data_viz.update_displays()
            
        # Update hardware status
        self.update_hardware_status()
        
        # Note: Inference now runs on its own dedicated timer when enabled
            
    def generate_synthetic_field_data(self):
        """Generate synthetic data to keep mystical field active when no RNG available"""
        try:
            import random
            import math
            
            # Generate 8 bytes of synthetic data with interesting patterns
            t = time.time()
            synthetic_rng = []
            
            for i in range(8):
                # Create complex patterns using multiple oscillators
                sine_component = int((math.sin(t * 0.3 + i * 0.5) + 1) * 127.5)
                chaos_component = random.randint(0, 255)
                noise_component = int((math.sin(t * 2.1 + i) * math.cos(t * 1.7 + i * 0.3) + 1) * 127.5)
                
                # Blend components for rich behavior
                value = (sine_component + chaos_component + noise_component) // 3
                synthetic_rng.append(value % 256)
            
            # Feed to mystical field
            synthetic_bytes = bytes(synthetic_rng)
            self.feed_rng_to_mystical_field(synthetic_bytes)
            
        except Exception as e:
            # Silent failure to avoid log spam
            pass
    
    def run_inference_cycle(self):
        """Run a single inference cycle for merged mode"""
        try:
            if hasattr(self, 'inference_engine') and self.inference_engine:
                # Trigger inference with current data
                self.inference_engine.predict_next()
                
        except Exception as e:
            self.log_message(f"⚠️ Error in inference cycle: {e}")
            
    def log_message(self, message: str):
        """Add message to event log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")


class MockInferenceEngine:
    """Simple mock inference engine for merged mode testing"""
    
    def __init__(self):
        self.active = True
    
    def stop(self):
        self.active = False


def create_painting_app(rng_device=None, eeg_bridge=None):
    """
    Create and return the painting application
    
    Args:
        rng_device: TrueRNG device instance
        eeg_bridge: EEG bridge instance
        
    Returns:
        Tuple of (app, window)
    """
    if not PYQT_AVAILABLE:
        raise ImportError("PyQt5 is required for the GUI. Install with: pip install PyQt5")
        
    app = QApplication(sys.argv)
    
    window = ConsciousnessMainWindow()
    if rng_device or eeg_bridge:
        window.set_hardware_devices(rng_device, eeg_bridge)
        
    return app, window


# Example usage
if __name__ == "__main__":
    if PYQT_AVAILABLE:
        app, window = create_painting_app()
        window.show()
        sys.exit(app.exec_())
    else:
        print("PyQt5 not available. Install with: pip install PyQt5")