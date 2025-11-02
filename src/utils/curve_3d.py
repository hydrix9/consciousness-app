"""
3D Curve Interpretation System

This module converts drawn lines into 3D curves representing interlocking
dials and circular patterns for the consciousness data generation system.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class CurveSegment:
    """A segment of a 3D curve"""
    points: List[Point3D]
    radius: float
    center: Point3D
    normal: Point3D  # Normal vector for the plane
    arc_angle: float  # Total arc angle in radians
    dial_id: int  # Which dial this segment belongs to


@dataclass
class InterlockingDial:
    """Represents a single interlocking dial"""
    dial_id: int
    center: Point3D
    radius: float
    normal: Point3D
    rotation_angle: float  # Current rotation
    segments: List[CurveSegment]
    color: Tuple[float, float, float, float]  # RGBA
    
    
class LineToCircleConverter:
    """Converts 2D drawing lines to 3D circular patterns"""
    
    def __init__(self):
        self.sensitivity = 1.0  # How sensitive the conversion is
        self.depth_range = (-5.0, 5.0)  # Z-axis range
        self.default_radius = 2.0
        self.curve_resolution = 50  # Points per curve segment
        
    def convert_stroke_to_curve(self, stroke_points: List[Tuple[float, float]], 
                               canvas_size: Tuple[int, int],
                               stroke_pressure: List[float] = None) -> CurveSegment:
        """
        Convert a 2D stroke into a 3D curve segment
        
        Args:
            stroke_points: List of (x, y) coordinates
            canvas_size: (width, height) of the canvas
            stroke_pressure: Optional pressure values for each point
            
        Returns:
            CurveSegment representing the 3D curve
        """
        if len(stroke_points) < 2:
            return None
            
        # Normalize coordinates to [-1, 1] range
        normalized_points = []
        for x, y in stroke_points:
            norm_x = (x / canvas_size[0]) * 2.0 - 1.0
            norm_y = -((y / canvas_size[1]) * 2.0 - 1.0)  # Flip Y axis
            normalized_points.append((norm_x, norm_y))
            
        # Analyze the stroke to determine circular properties
        circle_info = self._analyze_circular_pattern(normalized_points)
        
        # Generate 3D curve based on analysis
        curve_points = self._generate_3d_curve(normalized_points, circle_info, stroke_pressure)
        
        return CurveSegment(
            points=curve_points,
            radius=circle_info['radius'],
            center=circle_info['center'],
            normal=circle_info['normal'],
            arc_angle=circle_info['arc_angle'],
            dial_id=circle_info['dial_id']
        )
        
    def _analyze_circular_pattern(self, points: List[Tuple[float, float]]) -> Dict:
        """Analyze 2D points to extract circular pattern information"""
        
        # Calculate center of mass
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # Calculate average radius
        radii = []
        for x, y in points:
            radius = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            radii.append(radius)
            
        avg_radius = sum(radii) / len(radii) if radii else self.default_radius
        
        # Determine if it's a closed curve
        start_point = points[0]
        end_point = points[-1]
        is_closed = (abs(start_point[0] - end_point[0]) < 0.1 and 
                    abs(start_point[1] - end_point[1]) < 0.1)
        
        # Calculate arc angle
        if len(points) > 2:
            # Vector from center to first point
            start_vec = (points[0][0] - center_x, points[0][1] - center_y)
            end_vec = (points[-1][0] - center_x, points[-1][1] - center_y)
            
            # Calculate angle between vectors
            dot_product = start_vec[0] * end_vec[0] + start_vec[1] * end_vec[1]
            mag1 = math.sqrt(start_vec[0]**2 + start_vec[1]**2)
            mag2 = math.sqrt(end_vec[0]**2 + end_vec[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                arc_angle = math.acos(cos_angle)
            else:
                arc_angle = 0
                
            if is_closed:
                arc_angle = 2 * math.pi
        else:
            arc_angle = 0
            
        # Determine Z position and normal based on drawing characteristics
        z_offset = self._calculate_z_offset(points, avg_radius)
        normal = self._calculate_normal_vector(points, z_offset)
        
        # Assign dial ID based on spatial clustering
        dial_id = self._assign_dial_id(center_x, center_y, avg_radius)
        
        return {
            'center': Point3D(center_x, center_y, z_offset),
            'radius': avg_radius,
            'normal': normal,
            'arc_angle': arc_angle,
            'is_closed': is_closed,
            'dial_id': dial_id
        }
        
    def _calculate_z_offset(self, points: List[Tuple[float, float]], radius: float) -> float:
        """Calculate Z-axis offset based on drawing characteristics"""
        # Use radius and drawing complexity to determine depth
        complexity = len(points) / 100.0  # Normalize based on point count
        z_range = self.depth_range[1] - self.depth_range[0]
        
        # Map radius and complexity to Z position
        z_offset = (radius * complexity) * z_range + self.depth_range[0]
        z_offset = max(self.depth_range[0], min(self.depth_range[1], z_offset))
        
        return z_offset
        
    def _calculate_normal_vector(self, points: List[Tuple[float, float]], z_offset: float) -> Point3D:
        """Calculate normal vector for the curve plane"""
        if len(points) < 3:
            return Point3D(0, 0, 1)  # Default normal
            
        # Use first three points to calculate normal
        p1 = np.array([points[0][0], points[0][1], z_offset])
        p2 = np.array([points[1][0], points[1][1], z_offset])
        p3 = np.array([points[2][0], points[2][1], z_offset])
        
        # Calculate cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Normalize
        magnitude = np.linalg.norm(normal)
        if magnitude > 0:
            normal = normal / magnitude
        else:
            normal = np.array([0, 0, 1])
            
        return Point3D(normal[0], normal[1], normal[2])
        
    def _assign_dial_id(self, center_x: float, center_y: float, radius: float) -> int:
        """Assign a dial ID based on spatial clustering"""
        # Simple spatial hash for dial assignment
        grid_size = 0.5
        grid_x = int(center_x / grid_size)
        grid_y = int(center_y / grid_size)
        
        # Use spatial position to generate consistent dial ID
        dial_id = abs(hash((grid_x, grid_y))) % 8  # Max 8 interlocking dials
        return dial_id
        
    def _generate_3d_curve(self, points: List[Tuple[float, float]], 
                          circle_info: Dict, 
                          pressure: List[float] = None) -> List[Point3D]:
        """Generate smooth 3D curve points"""
        if not points:
            return []
            
        center = circle_info['center']
        radius = circle_info['radius']
        normal = circle_info['normal']
        
        # Create smooth interpolated curve
        curve_points = []
        
        if len(points) < 3:
            # Simple linear interpolation for short strokes
            for i, (x, y) in enumerate(points):
                z_variation = math.sin(i * math.pi / len(points)) * 0.1
                curve_points.append(Point3D(x, y, center.z + z_variation))
        else:
            # Spline interpolation for smooth curves
            t_values = np.linspace(0, 1, self.curve_resolution)
            
            for t in t_values:
                # Interpolate along the original points
                idx = int(t * (len(points) - 1))
                idx = min(idx, len(points) - 2)
                
                local_t = t * (len(points) - 1) - idx
                
                # Linear interpolation between adjacent points
                p1 = points[idx]
                p2 = points[idx + 1]
                
                x = p1[0] + (p2[0] - p1[0]) * local_t
                y = p1[1] + (p2[1] - p1[1]) * local_t
                
                # Add circular motion component
                angle = t * circle_info['arc_angle']
                circle_x = center.x + radius * math.cos(angle)
                circle_y = center.y + radius * math.sin(angle)
                
                # Blend linear interpolation with circular motion
                blend_factor = 0.3
                final_x = x * (1 - blend_factor) + circle_x * blend_factor
                final_y = y * (1 - blend_factor) + circle_y * blend_factor
                
                # Add Z variation based on circular motion
                z_variation = math.sin(angle * 2) * 0.2
                final_z = center.z + z_variation
                
                curve_points.append(Point3D(final_x, final_y, final_z))
                
        return curve_points


class InterlockingDialSystem:
    """Manages a system of interlocking dials"""
    
    def __init__(self):
        self.dials: Dict[int, InterlockingDial] = {}
        self.converter = LineToCircleConverter()
        self.next_dial_id = 0
        
        # Animation parameters
        self.rotation_speeds = {}  # dial_id -> rotation speed
        self.base_rotation_speed = 0.01  # radians per frame
        
    def add_stroke(self, stroke_points: List[Tuple[float, float]], 
                   canvas_size: Tuple[int, int],
                   color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                   pressure: List[float] = None) -> Optional[int]:
        """
        Add a new stroke to the dial system
        
        Args:
            stroke_points: List of (x, y) coordinates
            canvas_size: Canvas dimensions
            color: RGBA color tuple
            pressure: Optional pressure values
            
        Returns:
            Dial ID that the stroke was added to, or None if failed
        """
        curve_segment = self.converter.convert_stroke_to_curve(
            stroke_points, canvas_size, pressure
        )
        
        if not curve_segment:
            return None
            
        dial_id = curve_segment.dial_id
        
        # Create new dial if it doesn't exist
        if dial_id not in self.dials:
            self.dials[dial_id] = InterlockingDial(
                dial_id=dial_id,
                center=curve_segment.center,
                radius=curve_segment.radius,
                normal=curve_segment.normal,
                rotation_angle=0.0,
                segments=[],
                color=color
            )
            
            # Set rotation speed based on RNG data (if available)
            self.rotation_speeds[dial_id] = self.base_rotation_speed * (1 + dial_id * 0.1)
            
        # Add segment to dial
        self.dials[dial_id].segments.append(curve_segment)
        
        return dial_id
        
    def update_animation(self, dt: float, rng_data: List[float] = None, eeg_data: Dict = None):
        """
        Update dial animations based on time and data inputs
        
        Args:
            dt: Time delta in seconds
            rng_data: Random number data to influence animation
            eeg_data: EEG data to influence animation
        """
        for dial_id, dial in self.dials.items():
            base_speed = self.rotation_speeds.get(dial_id, self.base_rotation_speed)
            
            # Modify rotation speed based on data
            speed_modifier = 1.0
            
            if rng_data:
                # Use RNG data to create variation
                rng_influence = sum(rng_data[:4]) / 4.0  # Average of first 4 values
                speed_modifier *= (0.5 + rng_influence)
                
            if eeg_data:
                # Use EEG data to influence rotation (example: alpha waves)
                alpha_channels = ['O1', 'O2']  # Occipital channels for alpha
                alpha_values = []
                for channel in alpha_channels:
                    if channel in eeg_data:
                        alpha_values.append(abs(eeg_data[channel]))
                        
                if alpha_values:
                    eeg_influence = sum(alpha_values) / len(alpha_values) / 100.0  # Normalize
                    speed_modifier *= (0.8 + eeg_influence * 0.4)
                    
            # Update rotation
            dial.rotation_angle += base_speed * speed_modifier * dt
            dial.rotation_angle = dial.rotation_angle % (2 * math.pi)
            
    def get_dial_positions(self) -> Dict[int, Dict]:
        """Get current positions and orientations of all dials"""
        positions = {}
        
        for dial_id, dial in self.dials.items():
            positions[dial_id] = {
                'center': dial.center.to_tuple(),
                'radius': dial.radius,
                'rotation': dial.rotation_angle,
                'normal': dial.normal.to_tuple(),
                'color': dial.color,
                'segment_count': len(dial.segments)
            }
            
        return positions
        
    def export_to_json(self) -> str:
        """Export the dial system to JSON format"""
        export_data = {
            'dials': {},
            'metadata': {
                'dial_count': len(self.dials),
                'total_segments': sum(len(dial.segments) for dial in self.dials.values())
            }
        }
        
        for dial_id, dial in self.dials.items():
            dial_data = {
                'center': dial.center.to_tuple(),
                'radius': dial.radius,
                'normal': dial.normal.to_tuple(),
                'rotation_angle': dial.rotation_angle,
                'color': dial.color,
                'segments': []
            }
            
            for segment in dial.segments:
                segment_data = {
                    'points': [p.to_tuple() for p in segment.points],
                    'radius': segment.radius,
                    'center': segment.center.to_tuple(),
                    'arc_angle': segment.arc_angle
                }
                dial_data['segments'].append(segment_data)
                
            export_data['dials'][dial_id] = dial_data
            
        return json.dumps(export_data, indent=2)
        
    def clear_all_dials(self):
        """Clear all dials from the system"""
        self.dials.clear()
        self.rotation_speeds.clear()
        
    def get_interlocking_connections(self) -> List[Tuple[int, int]]:
        """
        Calculate which dials are interlocking based on proximity
        
        Returns:
            List of (dial_id1, dial_id2) tuples representing connections
        """
        connections = []
        dial_ids = list(self.dials.keys())
        
        for i, dial_id1 in enumerate(dial_ids):
            for dial_id2 in dial_ids[i+1:]:
                dial1 = self.dials[dial_id1]
                dial2 = self.dials[dial_id2]
                
                # Calculate distance between centers
                center1 = dial1.center.to_array()
                center2 = dial2.center.to_array()
                distance = np.linalg.norm(center2 - center1)
                
                # Check if dials are close enough to interlock
                combined_radius = dial1.radius + dial2.radius
                if distance < combined_radius * 1.2:  # 20% overlap threshold
                    connections.append((dial_id1, dial_id2))
                    
        return connections


# Example usage and testing
if __name__ == "__main__":
    # Create dial system
    dial_system = InterlockingDialSystem()
    
    # Simulate some drawing strokes
    canvas_size = (800, 600)
    
    # Circular stroke
    circle_points = []
    for i in range(50):
        angle = (i / 49.0) * 2 * math.pi
        x = 400 + 100 * math.cos(angle)
        y = 300 + 100 * math.sin(angle)
        circle_points.append((x, y))
        
    dial_id1 = dial_system.add_stroke(circle_points, canvas_size, (1.0, 0.0, 0.0, 1.0))
    
    # Spiral stroke
    spiral_points = []
    for i in range(30):
        angle = (i / 29.0) * 3 * math.pi
        radius = 50 + i * 2
        x = 200 + radius * math.cos(angle)
        y = 200 + radius * math.sin(angle)
        spiral_points.append((x, y))
        
    dial_id2 = dial_system.add_stroke(spiral_points, canvas_size, (0.0, 1.0, 0.0, 1.0))
    
    print(f"Created dials: {dial_id1}, {dial_id2}")
    
    # Simulate animation updates
    for frame in range(10):
        dt = 1.0 / 60.0  # 60 FPS
        rng_data = [0.3, 0.7, 0.2, 0.8]  # Mock RNG data
        eeg_data = {'O1': 15.0, 'O2': 12.0}  # Mock EEG data
        
        dial_system.update_animation(dt, rng_data, eeg_data)
        
        if frame % 5 == 0:  # Print every 5 frames
            positions = dial_system.get_dial_positions()
            for dial_id, pos in positions.items():
                print(f"Dial {dial_id}: rotation={pos['rotation']:.3f}")
                
    # Export to JSON
    json_export = dial_system.export_to_json()
    print(f"\nExported JSON (first 200 chars):\n{json_export[:200]}...")
    
    # Check interlocking
    connections = dial_system.get_interlocking_connections()
    print(f"\nInterlocking connections: {connections}")