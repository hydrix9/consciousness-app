"""
Utility functions for the consciousness application
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any


def normalize_coordinates(coords: List[Tuple[float, float]], 
                         canvas_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    Normalize canvas coordinates to [-1, 1] range
    
    Args:
        coords: List of (x, y) coordinates
        canvas_size: (width, height) of canvas
        
    Returns:
        Normalized coordinates
    """
    width, height = canvas_size
    normalized = []
    
    for x, y in coords:
        norm_x = (x / width) * 2.0 - 1.0
        norm_y = -((y / height) * 2.0 - 1.0)  # Flip Y axis
        normalized.append((norm_x, norm_y))
        
    return normalized


def denormalize_coordinates(coords: List[Tuple[float, float]], 
                           canvas_size: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    Convert normalized coordinates back to canvas coordinates
    
    Args:
        coords: List of normalized (x, y) coordinates in [-1, 1] range
        canvas_size: (width, height) of canvas
        
    Returns:
        Canvas coordinates
    """
    width, height = canvas_size
    denormalized = []
    
    for norm_x, norm_y in coords:
        x = ((norm_x + 1.0) / 2.0) * width
        y = ((-norm_y + 1.0) / 2.0) * height
        denormalized.append((x, y))
        
    return denormalized


def calculate_stroke_features(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate features of a drawing stroke
    
    Args:
        points: List of stroke points
        
    Returns:
        Dictionary of stroke features
    """
    if len(points) < 2:
        return {}
        
    # Calculate total length
    total_length = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        total_length += np.sqrt(dx*dx + dy*dy)
        
    # Calculate bounding box
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate curvature (simplified)
    curvature = 0.0
    if len(points) >= 3:
        angles = []
        for i in range(1, len(points) - 1):
            v1 = (points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
            v2 = (points[i+1][0] - points[i][0], points[i+1][1] - points[i][1])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = np.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
            mag2 = np.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.arccos(cos_angle)
                angles.append(angle)
                
        curvature = np.mean(angles) if angles else 0.0
        
    return {
        'length': total_length,
        'width': max_x - min_x,
        'height': max_y - min_y,
        'curvature': curvature,
        'point_count': len(points),
        'complexity': total_length / len(points) if len(points) > 0 else 0
    }


def smooth_stroke(points: List[Tuple[float, float]], 
                 smoothing_factor: float = 0.5) -> List[Tuple[float, float]]:
    """
    Apply smoothing to a stroke
    
    Args:
        points: Original stroke points
        smoothing_factor: Amount of smoothing (0 = no smoothing, 1 = maximum)
        
    Returns:
        Smoothed stroke points
    """
    if len(points) < 3:
        return points
        
    smoothed = [points[0]]  # Keep first point
    
    for i in range(1, len(points) - 1):
        # Apply weighted average with neighbors
        prev_point = points[i-1]
        curr_point = points[i]
        next_point = points[i+1]
        
        smooth_x = (1 - smoothing_factor) * curr_point[0] + \
                  smoothing_factor * 0.5 * (prev_point[0] + next_point[0])
        smooth_y = (1 - smoothing_factor) * curr_point[1] + \
                  smoothing_factor * 0.5 * (prev_point[1] + next_point[1])
                  
        smoothed.append((smooth_x, smooth_y))
        
    smoothed.append(points[-1])  # Keep last point
    
    return smoothed


def interpolate_stroke(points: List[Tuple[float, float]], 
                      target_length: int) -> List[Tuple[float, float]]:
    """
    Interpolate stroke to have target number of points
    
    Args:
        points: Original stroke points
        target_length: Desired number of points
        
    Returns:
        Interpolated stroke with target_length points
    """
    if len(points) <= 1:
        return points
        
    if len(points) == target_length:
        return points
        
    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        dist = np.sqrt(dx*dx + dy*dy)
        distances.append(distances[-1] + dist)
        
    total_length = distances[-1]
    if total_length == 0:
        return points
        
    # Interpolate points at regular intervals
    interpolated = []
    for i in range(target_length):
        target_dist = (i / (target_length - 1)) * total_length
        
        # Find the two points to interpolate between
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist <= distances[j + 1]:
                # Linear interpolation
                if distances[j + 1] == distances[j]:
                    t = 0
                else:
                    t = (target_dist - distances[j]) / (distances[j + 1] - distances[j])
                    
                x = points[j][0] + t * (points[j + 1][0] - points[j][0])
                y = points[j][1] + t * (points[j + 1][1] - points[j][1])
                interpolated.append((x, y))
                break
        else:
            # Fallback to last point
            interpolated.append(points[-1])
            
    return interpolated


def timestamp_to_string(timestamp: float) -> str:
    """Convert timestamp to readable string"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def create_session_summary(session_data: Dict[str, Any]) -> str:
    """Create a human-readable session summary"""
    metadata = session_data.get('metadata', {})
    
    summary = f"""
CONSCIOUSNESS SESSION SUMMARY
=============================

Session ID: {metadata.get('session_id', 'Unknown')}
Start Time: {timestamp_to_string(metadata.get('start_time', 0))}
Duration: {metadata.get('end_time', 0) - metadata.get('start_time', 0):.1f} seconds

Hardware Configuration:
  Drawing Delay Offset: {metadata.get('drawing_delay_offset', 0)} seconds

Data Collected:
  Drawing Actions: {metadata.get('total_drawing_actions', 0)}
  RNG Samples: {metadata.get('total_rng_samples', 0)}
  EEG Samples: {metadata.get('total_eeg_samples', 0)}

Participant: {metadata.get('participant_id', 'Anonymous')}
Notes: {metadata.get('experiment_notes', 'None')}
"""
    
    return summary


def validate_data_integrity(session_data: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate the integrity of session data
    
    Args:
        session_data: Session data dictionary
        
    Returns:
        Dictionary of validation results
    """
    results = {
        'has_metadata': 'metadata' in session_data,
        'has_rng_data': 'rng_data' in session_data and len(session_data['rng_data']) > 0,
        'has_eeg_data': 'eeg_data' in session_data and len(session_data['eeg_data']) > 0,
        'has_drawing_data': 'drawing_data' in session_data and len(session_data['drawing_data']) > 0,
        'timestamps_valid': True,
        'data_synchronized': True
    }
    
    # Check timestamp validity
    if results['has_rng_data']:
        rng_timestamps = [sample.get('timestamp', 0) for sample in session_data['rng_data']]
        results['timestamps_valid'] = all(t > 0 for t in rng_timestamps)
        
    # Check data synchronization (simplified)
    if results['has_drawing_data'] and results['has_rng_data']:
        drawing_times = [action.get('timestamp', 0) for action in session_data['drawing_data']]
        rng_times = [sample.get('timestamp', 0) for sample in session_data['rng_data']]
        
        if drawing_times and rng_times:
            drawing_range = (min(drawing_times), max(drawing_times))
            rng_range = (min(rng_times), max(rng_times))
            
            # Check if time ranges overlap
            overlap = not (drawing_range[1] < rng_range[0] or rng_range[1] < drawing_range[0])
            results['data_synchronized'] = overlap
            
    return results