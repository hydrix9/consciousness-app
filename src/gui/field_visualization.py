"""
Mystical Field Visualization

Creates a physical particle field that responds to consciousness RNG streams.
Particles exhibit emergent behaviors with occult-themed visual effects.
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

try:
    from PyQt5.QtWidgets import QWidget
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QRadialGradient
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@dataclass
class Particle:
    """A mystical particle with physical properties"""
    pos: np.ndarray  # [x, y] position
    vel: np.ndarray  # [x, y] velocity 
    mass: float      # affects attraction and inertia
    charge: float    # affects inter-particle forces
    energy: float    # visual intensity/glow
    age: int         # frames since creation
    max_age: int     # lifespan before fade
    trail: deque     # position history for trails


class MysticalField:
    """Physical field simulation driven by RNG consciousness stream"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.max_particles = 150
        self.rng_buffer = deque(maxlen=64)  # Store recent RNG bytes
        
        # Physics constants
        self.gravity_strength = 0.02
        self.attraction_strength = 0.8
        self.repulsion_strength = 0.3
        self.damping = 0.98
        self.jitter_strength = 0.5
        
        # Visual settings
        self.trail_length = 20
        self.base_energy = 0.7
        self.energy_decay = 0.995
        
        # Occult color palette
        self.colors = {
            'energy': QColor(138, 43, 226),    # Deep purple
            'mystical': QColor(75, 0, 130),    # Indigo
            'ethereal': QColor(255, 20, 147),  # Deep pink
            'cosmic': QColor(0, 255, 255),     # Cyan
            'shadow': QColor(72, 61, 139),     # Dark slate blue
            'gold': QColor(255, 215, 0),       # Mystical gold
        }
        
        # Initialize some seed particles
        self._create_initial_particles()
    
    def _create_initial_particles(self):
        """Create initial mystical particle distribution"""
        for i in range(30):
            # Create particles in sacred geometry patterns
            angle = (i / 30) * 2 * math.pi
            radius = 100 + 50 * math.sin(angle * 3)  # Flower of life pattern
            
            x = self.width / 2 + radius * math.cos(angle)
            y = self.height / 2 + radius * math.sin(angle)
            
            particle = Particle(
                pos=np.array([x, y], dtype=float),
                vel=np.array([0.0, 0.0]),
                mass=0.5 + np.random.random() * 0.5,
                charge=np.random.choice([-1, 1]) * (0.3 + np.random.random() * 0.7),
                energy=self.base_energy + np.random.random() * 0.3,
                age=0,
                max_age=300 + np.random.randint(200),
                trail=deque(maxlen=self.trail_length)
            )
            self.particles.append(particle)
    
    def add_rng_data(self, rng_bytes: bytes):
        """Feed RNG stream into the field"""
        for byte in rng_bytes:
            self.rng_buffer.append(byte)
    
    def _get_rng_influence(self) -> Tuple[float, float, float]:
        """Extract mystical influences from RNG stream"""
        if len(self.rng_buffer) < 8:
            return 0.0, 0.0, 0.0
        
        # Use recent RNG bytes to influence field
        recent = list(self.rng_buffer)[-8:]
        
        # Jitter strength from entropy
        entropy = np.std(recent) / 255.0
        jitter = entropy * self.jitter_strength
        
        # Force direction from byte patterns
        force_x = (sum(recent[::2]) - sum(recent[1::2])) / (4 * 255.0)
        force_y = (sum(recent[:4]) - sum(recent[4:])) / (4 * 255.0)
        
        return jitter, force_x, force_y
    
    def _apply_consciousness_forces(self, particle: Particle, jitter: float, force_x: float, force_y: float):
        """Apply RNG-driven mystical forces to particle"""
        # Random jitter based on consciousness entropy
        if len(self.rng_buffer) >= 2:
            rng_x = (self.rng_buffer[-1] - 128) / 128.0
            rng_y = (self.rng_buffer[-2] - 128) / 128.0
            
            particle.vel[0] += rng_x * jitter
            particle.vel[1] += rng_y * jitter
        
        # Directional force from consciousness patterns
        particle.vel[0] += force_x * 0.1
        particle.vel[1] += force_y * 0.1
        
        # Energy modulation based on RNG coherence
        if len(self.rng_buffer) >= 4:
            coherence = 1.0 - (np.std(list(self.rng_buffer)[-4:]) / 255.0)
            particle.energy += coherence * 0.02 - 0.01
            particle.energy = max(0.1, min(1.5, particle.energy))
    
    def _apply_inter_particle_forces(self, particle: Particle):
        """Apply mystical attraction and repulsion between particles"""
        for other in self.particles:
            if other is particle:
                continue
            
            # Calculate distance and direction
            dx = other.pos[0] - particle.pos[0]
            dy = other.pos[1] - particle.pos[1]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < 1.0:  # Avoid division by zero
                dist_sq = 1.0
            
            dist = math.sqrt(dist_sq)
            
            # Normalize direction
            if dist > 0:
                dx /= dist
                dy /= dist
            
            # Mystical forces based on charge and mass
            if particle.charge * other.charge < 0:  # Opposite charges attract
                force_mag = (self.attraction_strength * abs(particle.charge * other.charge)) / dist_sq
                particle.vel[0] += dx * force_mag
                particle.vel[1] += dy * force_mag
            else:  # Same charges repel
                force_mag = (self.repulsion_strength * abs(particle.charge * other.charge)) / dist_sq
                particle.vel[0] -= dx * force_mag
                particle.vel[1] -= dy * force_mag
            
            # Weak gravitational attraction for clustering
            grav_force = (self.gravity_strength * particle.mass * other.mass) / dist_sq
            particle.vel[0] += dx * grav_force
            particle.vel[1] += dy * grav_force
    
    def _apply_boundary_forces(self, particle: Particle):
        """Apply mystical boundary effects"""
        margin = 50
        
        # Soft boundaries with ethereal reflection
        if particle.pos[0] < margin:
            particle.vel[0] += (margin - particle.pos[0]) * 0.01
        elif particle.pos[0] > self.width - margin:
            particle.vel[0] -= (particle.pos[0] - (self.width - margin)) * 0.01
            
        if particle.pos[1] < margin:
            particle.vel[1] += (margin - particle.pos[1]) * 0.01
        elif particle.pos[1] > self.height - margin:
            particle.vel[1] -= (particle.pos[1] - (self.height - margin)) * 0.01
    
    def update(self):
        """Update the mystical field physics"""
        if not self.particles:
            return
        
        # Get consciousness influences from RNG
        jitter, force_x, force_y = self._get_rng_influence()
        
        # Create a list of particle indices to remove (avoid modifying list during iteration)
        particles_to_remove = []
        
        # Update each particle
        for i, particle in enumerate(self.particles):
            # Store position in trail
            particle.trail.append((particle.pos[0], particle.pos[1]))
            
            # Apply various mystical forces
            self._apply_consciousness_forces(particle, jitter, force_x, force_y)
            self._apply_inter_particle_forces(particle)
            self._apply_boundary_forces(particle)
            
            # Update position with velocity
            particle.pos += particle.vel
            
            # Apply damping
            particle.vel *= self.damping
            
            # Update energy and age
            particle.energy *= self.energy_decay
            particle.age += 1
            
            # Mark old particles for removal by index
            if particle.age > particle.max_age or particle.energy < 0.05:
                particles_to_remove.append(i)
        
        # Remove old particles safely (reverse order to maintain indices)
        for i in reversed(particles_to_remove):
            del self.particles[i]
        
        # Occasionally spawn new particles from consciousness
        if len(self.rng_buffer) >= 8 and len(self.particles) < self.max_particles:
            if np.random.random() < 0.02:  # 2% chance per frame
                self._spawn_consciousness_particle()
    
    def _spawn_consciousness_particle(self):
        """Spawn a new particle influenced by consciousness"""
        if len(self.rng_buffer) < 8:
            return
        
        # Use RNG to determine spawn location
        rng_bytes = list(self.rng_buffer)[-8:]
        
        x = (rng_bytes[0] / 255.0) * self.width
        y = (rng_bytes[1] / 255.0) * self.height
        
        # Bias towards center with mystical spiral
        center_x, center_y = self.width / 2, self.height / 2
        angle = (rng_bytes[2] / 255.0) * 2 * math.pi
        radius = (rng_bytes[3] / 255.0) * 150
        
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        # Create particle with RNG-influenced properties
        particle = Particle(
            pos=np.array([x, y], dtype=float),
            vel=np.array([
                (rng_bytes[4] - 128) / 64.0,
                (rng_bytes[5] - 128) / 64.0
            ]),
            mass=0.3 + (rng_bytes[6] / 255.0) * 0.7,
            charge=np.random.choice([-1, 1]) * (0.5 + (rng_bytes[7] / 255.0) * 0.5),
            energy=self.base_energy + (np.std(rng_bytes) / 255.0) * 0.5,
            age=0,
            max_age=200 + np.random.randint(300),
            trail=deque(maxlen=self.trail_length)
        )
        
        self.particles.append(particle)


class MysticalFieldWidget(QWidget):
    """Qt widget for rendering the mystical field"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.field = MysticalField(800, 600)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_and_repaint)
        self.timer.start(16)  # ~60 FPS
        
        # Background
        self.setStyleSheet("background-color: black;")
    
    def _update_and_repaint(self):
        """Update field and trigger repaint"""
        self.field.update()
        self.update()
    
    def add_rng_data(self, rng_bytes: bytes):
        """Add RNG data to influence the field"""
        self.field.add_rng_data(rng_bytes)
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        self.field.width = event.size().width()
        self.field.height = event.size().height()
    
    def paintEvent(self, event):
        """Render the mystical field"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background with subtle mystical gradient
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(0, 0, 20))      # Deep space blue
        gradient.setColorAt(0.5, QColor(10, 0, 30))   # Mystical purple
        gradient.setColorAt(1, QColor(20, 0, 20))     # Dark magenta
        
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw particle trails first (behind particles)
        self._draw_trails(painter)
        
        # Draw mystical connections between nearby particles
        self._draw_connections(painter)
        
        # Draw particles with mystical glow
        self._draw_particles(painter)
    
    def _draw_trails(self, painter):
        """Draw ethereal particle trails"""
        for particle in self.field.particles:
            if len(particle.trail) < 2:
                continue
            
            # Create fading trail
            trail_points = list(particle.trail)
            for i in range(len(trail_points) - 1):
                alpha = int(255 * particle.energy * (i / len(trail_points)) * 0.3)
                
                # Color based on charge
                if particle.charge > 0:
                    color = QColor(138, 43, 226, alpha)  # Purple for positive
                else:
                    color = QColor(0, 255, 255, alpha)   # Cyan for negative
                
                pen = QPen(color, 1.5)
                painter.setPen(pen)
                
                x1, y1 = trail_points[i]
                x2, y2 = trail_points[i + 1]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def _draw_connections(self, painter):
        """Draw mystical connections between nearby particles"""
        connection_distance = 120
        
        for i, particle1 in enumerate(self.field.particles):
            for particle2 in self.field.particles[i+1:]:
                dx = particle2.pos[0] - particle1.pos[0]
                dy = particle2.pos[1] - particle1.pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < connection_distance:
                    # Connection strength based on distance and energy
                    strength = (1.0 - dist / connection_distance) * min(particle1.energy, particle2.energy)
                    alpha = int(255 * strength * 0.2)
                    
                    # Color based on charge interaction
                    if particle1.charge * particle2.charge < 0:
                        color = QColor(255, 215, 0, alpha)  # Gold for attraction
                    else:
                        color = QColor(255, 20, 147, alpha)  # Pink for repulsion
                    
                    pen = QPen(color, 1.0)
                    painter.setPen(pen)
                    painter.drawLine(int(particle1.pos[0]), int(particle1.pos[1]), 
                                   int(particle2.pos[0]), int(particle2.pos[1]))
    
    def _draw_particles(self, painter):
        """Draw particles with mystical glow effects"""
        for particle in self.field.particles:
            x, y = int(particle.pos[0]), int(particle.pos[1])
            
            # Base size based on mass and energy
            base_size = 3 + particle.mass * 4 + particle.energy * 3
            
            # Pulsing effect based on age
            pulse = 1.0 + 0.3 * math.sin(particle.age * 0.2)
            size = int(base_size * pulse)
            
            # Color based on charge and energy
            if particle.charge > 0:
                # Positive particles: Purple to gold spectrum
                r = int(138 + particle.energy * 117)  # 138->255
                g = int(43 + particle.energy * 172)   # 43->215
                b = int(226 - particle.energy * 226)  # 226->0
            else:
                # Negative particles: Cyan to blue spectrum
                r = int(particle.energy * 100)       # 0->100
                g = int(255 * particle.energy)       # Full cyan
                b = 255
            
            alpha = int(255 * particle.energy)
            
            # Draw glow effect
            glow_size = int(size * 2.5)
            glow_gradient = QRadialGradient(x, y, glow_size)
            glow_gradient.setColorAt(0, QColor(r, g, b, alpha // 3))
            glow_gradient.setColorAt(1, QColor(r, g, b, 0))
            
            painter.setBrush(QBrush(glow_gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - glow_size, y - glow_size, 
                              glow_size * 2, glow_size * 2)
            
            # Draw core particle
            core_gradient = QRadialGradient(x, y, size)
            core_gradient.setColorAt(0, QColor(255, 255, 255, alpha))
            core_gradient.setColorAt(0.7, QColor(r, g, b, alpha))
            core_gradient.setColorAt(1, QColor(r//2, g//2, b//2, alpha//2))
            
            painter.setBrush(QBrush(core_gradient))
            painter.drawEllipse(x - size, y - size, size * 2, size * 2)


def create_mystical_field_widget(parent=None) -> Optional[MysticalFieldWidget]:
    """Factory function to create mystical field widget"""
    if not PYQT_AVAILABLE:
        return None
    
    return MysticalFieldWidget(parent)


if __name__ == "__main__":
    # Test the mystical field
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Mystical Consciousness Field")
    window.setGeometry(100, 100, 900, 700)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    layout = QVBoxLayout(central_widget)
    
    field_widget = MysticalFieldWidget()
    layout.addWidget(field_widget)
    
    # Simulate RNG data
    import random
    def add_test_data():
        rng_bytes = bytes([random.randint(0, 255) for _ in range(8)])
        field_widget.add_rng_data(rng_bytes)
    
    test_timer = QTimer()
    test_timer.timeout.connect(add_test_data)
    test_timer.start(100)  # Add test data every 100ms
    
    window.show()
    sys.exit(app.exec_())