"""
Real Data Loader for Training

This module loads and preprocesses real consciousness session data for ML training.
It handles HDF5 files produced by the data logging system and converts them into
training-ready formats.
"""

import os
import json
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class RealSessionData:
    """Real session data loaded from files"""
    session_id: str
    metadata: Dict[str, Any]
    rng_timestamps: np.ndarray
    rng_values: np.ndarray
    eeg_timestamps: Optional[np.ndarray]
    eeg_channels: Optional[Dict[str, np.ndarray]]
    drawing_timestamps: Optional[np.ndarray]
    drawing_actions: Optional[Dict[str, np.ndarray]]
    dial_timestamps: Optional[np.ndarray]
    dial_positions: Optional[List[Dict]]
    
    
class RealDataLoader:
    """Loads and preprocesses real consciousness session data"""
    
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the real data loader
        
        Args:
            data_directory: Directory containing session data files
        """
        self.data_directory = data_directory
        self.logger = logging.getLogger(__name__)
        
    def find_session_files(self, file_extension: str = ".h5") -> List[str]:
        """
        Find all session files in the data directory
        
        Args:
            file_extension: File extension to search for (.h5, .json)
            
        Returns:
            List of file paths
        """
        session_files = []
        
        if not os.path.exists(self.data_directory):
            self.logger.warning(f"Data directory not found: {self.data_directory}")
            return session_files
            
        for filename in os.listdir(self.data_directory):
            if filename.endswith(file_extension) and 'session_' in filename and 'error' not in filename:
                filepath = os.path.join(self.data_directory, filename)
                session_files.append(filepath)
                
        self.logger.info(f"Found {len(session_files)} session files")
        return sorted(session_files)
    
    def load_session_from_hdf5(self, filepath: str) -> Optional[RealSessionData]:
        """
        Load a single session from HDF5 file
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            RealSessionData object or None if loading fails
        """
        try:
            with h5py.File(filepath, 'r') as h5f:
                # Extract session ID from filename
                session_id = os.path.basename(filepath).replace('.h5', '')
                
                # Load metadata
                metadata = {}
                if 'metadata' in h5f:
                    for key, value in h5f['metadata'].attrs.items():
                        if isinstance(value, bytes):
                            try:
                                # Try to decode as JSON first
                                decoded = value.decode('utf-8')
                                if decoded.startswith('{') or decoded.startswith('['):
                                    metadata[key] = json.loads(decoded)
                                else:
                                    metadata[key] = decoded
                            except:
                                metadata[key] = value.decode('utf-8', errors='replace')
                        else:
                            metadata[key] = value
                
                # Load RNG data
                rng_timestamps = None
                rng_values = None
                if 'rng_data' in h5f:
                    rng_group = h5f['rng_data']
                    if 'timestamps' in rng_group:
                        rng_timestamps = rng_group['timestamps'][:]
                    if 'values' in rng_group:
                        rng_values = rng_group['values'][:]
                
                # Load EEG data
                eeg_timestamps = None
                eeg_channels = None
                if 'eeg_data' in h5f:
                    eeg_group = h5f['eeg_data']
                    if 'timestamps' in eeg_group:
                        eeg_timestamps = eeg_group['timestamps'][:]
                    
                    # Load EEG channels
                    eeg_channels = {}
                    for key in eeg_group.keys():
                        if key.startswith('channel_'):
                            channel_name = key.replace('channel_', '')
                            eeg_channels[channel_name] = eeg_group[key][:]
                
                # Load drawing data
                drawing_timestamps = None
                drawing_actions = None
                if 'drawing_data' in h5f:
                    drawing_group = h5f['drawing_data']
                    drawing_actions = {}
                    
                    # Load individual fields
                    for field_name in ['timestamps', 'action_types', 'positions_x', 'positions_y',
                                      'colors_r', 'colors_g', 'colors_b', 'colors_a',
                                      'brush_sizes', 'pressures', 'consciousness_layers', 
                                      'pocket_dimensions', 'metadata']:
                        if field_name in drawing_group:
                            data = drawing_group[field_name][:]
                            if field_name == 'action_types':
                                # Decode byte strings to regular strings
                                data = [item.decode('ascii', errors='replace') if isinstance(item, bytes) else str(item) for item in data]
                            elif field_name == 'metadata':
                                # Decode JSON metadata
                                decoded_metadata = []
                                for item in data:
                                    try:
                                        if isinstance(item, bytes):
                                            json_str = item.decode('ascii', errors='replace')
                                            decoded_metadata.append(json.loads(json_str))
                                        else:
                                            decoded_metadata.append({})
                                    except:
                                        decoded_metadata.append({})
                                data = decoded_metadata
                            drawing_actions[field_name] = data
                    
                    if 'timestamps' in drawing_actions:
                        drawing_timestamps = np.array(drawing_actions['timestamps'])
                
                # Load dial data
                dial_timestamps = None
                dial_positions = None
                if 'dial_data' in h5f:
                    dial_group = h5f['dial_data']
                    if 'timestamps' in dial_group:
                        dial_timestamps = dial_group['timestamps'][:]
                    if 'positions' in dial_group:
                        # Decode JSON position data
                        position_data = dial_group['positions'][:]
                        dial_positions = []
                        for item in position_data:
                            try:
                                if isinstance(item, bytes):
                                    json_str = item.decode('ascii', errors='replace')
                                    dial_positions.append(json.loads(json_str))
                                else:
                                    dial_positions.append({})
                            except:
                                dial_positions.append({})
                
                return RealSessionData(
                    session_id=session_id,
                    metadata=metadata,
                    rng_timestamps=rng_timestamps,
                    rng_values=rng_values,
                    eeg_timestamps=eeg_timestamps,
                    eeg_channels=eeg_channels,
                    drawing_timestamps=drawing_timestamps,
                    drawing_actions=drawing_actions,
                    dial_timestamps=dial_timestamps,
                    dial_positions=dial_positions
                )
                
        except Exception as e:
            self.logger.error(f"Error loading session from {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_session_from_json(self, filepath: str) -> Optional[RealSessionData]:
        """
        Load a single session from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            RealSessionData object or None if loading fails
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            session_id = os.path.basename(filepath).replace('.json', '')
            
            # Extract data components
            metadata = data.get('metadata', {})
            
            # RNG data
            rng_data = data.get('rng_data', [])
            rng_timestamps = np.array([item['timestamp'] for item in rng_data]) if rng_data else None
            rng_values = np.array([item['normalized'] for item in rng_data]) if rng_data else None
            
            # EEG data
            eeg_data = data.get('eeg_data', [])
            eeg_timestamps = np.array([item['timestamp'] for item in eeg_data]) if eeg_data else None
            eeg_channels = None
            if eeg_data:
                # Extract channel names
                channel_names = list(eeg_data[0]['channels'].keys()) if eeg_data else []
                eeg_channels = {}
                for channel in channel_names:
                    eeg_channels[channel] = np.array([item['channels'][channel] for item in eeg_data])
            
            # Drawing data
            drawing_data = data.get('drawing_data', [])
            drawing_timestamps = np.array([item['timestamp'] for item in drawing_data]) if drawing_data else None
            drawing_actions = None
            if drawing_data:
                drawing_actions = {
                    'timestamps': [item['timestamp'] for item in drawing_data],
                    'action_types': [item['action_type'] for item in drawing_data],
                    'positions_x': [item['position'][0] for item in drawing_data],
                    'positions_y': [item['position'][1] for item in drawing_data],
                    'colors_r': [item['color'][0] for item in drawing_data],
                    'colors_g': [item['color'][1] for item in drawing_data],
                    'colors_b': [item['color'][2] for item in drawing_data],
                    'colors_a': [item['color'][3] for item in drawing_data],
                    'brush_sizes': [item['brush_size'] for item in drawing_data],
                    'pressures': [item['pressure'] for item in drawing_data],
                }
                
                # Add enhanced fields if present
                if drawing_data and 'consciousness_layer' in drawing_data[0]:
                    drawing_actions['consciousness_layers'] = [item.get('consciousness_layer', 1) for item in drawing_data]
                if drawing_data and 'pocket_dimension' in drawing_data[0]:
                    drawing_actions['pocket_dimensions'] = [item.get('pocket_dimension', 1) for item in drawing_data]
                if drawing_data and 'metadata' in drawing_data[0]:
                    drawing_actions['metadata'] = [item.get('metadata', {}) for item in drawing_data]
            
            # Dial data
            dial_data = data.get('dial_data', [])
            dial_timestamps = np.array([item['timestamp'] for item in dial_data]) if dial_data else None
            dial_positions = [item['positions'] for item in dial_data] if dial_data else None
            
            return RealSessionData(
                session_id=session_id,
                metadata=metadata,
                rng_timestamps=rng_timestamps,
                rng_values=rng_values,
                eeg_timestamps=eeg_timestamps,
                eeg_channels=eeg_channels,
                drawing_timestamps=drawing_timestamps,
                drawing_actions=drawing_actions,
                dial_timestamps=dial_timestamps,
                dial_positions=dial_positions
            )
            
        except Exception as e:
            self.logger.error(f"Error loading session from {filepath}: {e}")
            return None
    
    def load_multiple_sessions(self, file_paths: List[str] = None) -> List[RealSessionData]:
        """
        Load multiple sessions from files
        
        Args:
            file_paths: List of file paths. If None, loads all found session files
            
        Returns:
            List of RealSessionData objects
        """
        if file_paths is None:
            # Try HDF5 files first, then JSON
            file_paths = self.find_session_files(".h5")
            if not file_paths:
                file_paths = self.find_session_files(".json")
        
        sessions = []
        for filepath in file_paths:
            self.logger.info(f"Loading session from: {filepath}")
            
            if filepath.endswith('.h5'):
                session = self.load_session_from_hdf5(filepath)
            elif filepath.endswith('.json'):
                session = self.load_session_from_json(filepath)
            else:
                self.logger.warning(f"Unsupported file format: {filepath}")
                continue
            
            if session:
                sessions.append(session)
                self.logger.info(f"Successfully loaded session: {session.session_id}")
            else:
                self.logger.warning(f"Failed to load session from: {filepath}")
        
        self.logger.info(f"Loaded {len(sessions)} sessions total")
        return sessions
    
    def combine_sessions(self, sessions: List[RealSessionData]) -> Dict[str, np.ndarray]:
        """
        Combine multiple sessions into a single dataset
        
        Args:
            sessions: List of RealSessionData objects
            
        Returns:
            Combined dataset with synchronized timestamps
        """
        if not sessions:
            return {}
        
        # Combine all data
        all_rng_timestamps = []
        all_rng_values = []
        all_eeg_timestamps = []
        all_eeg_data = {}
        all_drawing_timestamps = []
        all_drawing_data = {}
        all_dial_timestamps = []
        all_dial_data = []
        
        for session in sessions:
            # RNG data
            if session.rng_timestamps is not None and session.rng_values is not None:
                all_rng_timestamps.extend(session.rng_timestamps)
                all_rng_values.extend(session.rng_values)
            
            # EEG data
            if session.eeg_timestamps is not None and session.eeg_channels:
                all_eeg_timestamps.extend(session.eeg_timestamps)
                for channel, data in session.eeg_channels.items():
                    if channel not in all_eeg_data:
                        all_eeg_data[channel] = []
                    all_eeg_data[channel].extend(data)
            
            # Drawing data
            if session.drawing_timestamps is not None and session.drawing_actions:
                all_drawing_timestamps.extend(session.drawing_timestamps)
                for field, data in session.drawing_actions.items():
                    if field not in all_drawing_data:
                        all_drawing_data[field] = []
                    all_drawing_data[field].extend(data)
            
            # Dial data
            if session.dial_timestamps is not None and session.dial_positions:
                all_dial_timestamps.extend(session.dial_timestamps)
                all_dial_data.extend(session.dial_positions)
        
        # Create combined dataset
        combined = {}
        
        if all_rng_timestamps:
            combined['rng_timestamps'] = np.array(all_rng_timestamps)
            combined['rng_values'] = np.array(all_rng_values)
        
        if all_eeg_timestamps:
            combined['eeg_timestamps'] = np.array(all_eeg_timestamps)
            for channel, data in all_eeg_data.items():
                combined[f'eeg_{channel}'] = np.array(data)
        
        if all_drawing_timestamps:
            combined['drawing_timestamps'] = np.array(all_drawing_timestamps)
            for field, data in all_drawing_data.items():
                if field != 'timestamps':  # Avoid duplicate timestamps
                    combined[f'drawing_{field}'] = np.array(data)
        
        if all_dial_timestamps:
            combined['dial_timestamps'] = np.array(all_dial_timestamps)
            combined['dial_positions'] = all_dial_data
        
        self.logger.info(f"Combined dataset keys: {list(combined.keys())}")
        return combined
    
    def prepare_training_data(self, sessions: List[RealSessionData], 
                            sequence_length: int = 100,
                            prediction_horizon: int = 1) -> Dict[str, np.ndarray]:
        """
        Prepare real session data for ML training
        
        Args:
            sessions: List of loaded sessions
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            Training data dictionary with inputs and targets
        """
        # Combine all sessions
        combined_data = self.combine_sessions(sessions)
        
        if not combined_data:
            self.logger.warning("No data available for training preparation")
            return {}
        
        # Synchronize data by timestamp (simplified approach)
        training_data = self._synchronize_for_training(combined_data, sequence_length, prediction_horizon)
        
        return training_data
    
    def _synchronize_for_training(self, combined_data: Dict[str, np.ndarray], 
                                 sequence_length: int, prediction_horizon: int) -> Dict[str, np.ndarray]:
        """
        Synchronize combined data for training
        
        Args:
            combined_data: Combined session data
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            
        Returns:
            Synchronized training data
        """
        # Find common time range
        all_timestamps = []
        
        if 'rng_timestamps' in combined_data:
            all_timestamps.extend(combined_data['rng_timestamps'])
        if 'eeg_timestamps' in combined_data:
            all_timestamps.extend(combined_data['eeg_timestamps'])
        if 'drawing_timestamps' in combined_data:
            all_timestamps.extend(combined_data['drawing_timestamps'])
        
        if not all_timestamps:
            return {}
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Create regular time grid (100 Hz)
        sample_rate = 100.0
        time_grid = np.arange(min_time, max_time, 1.0 / sample_rate)
        
        # Interpolate all data onto common grid
        interpolated = {'timestamps': time_grid}
        
        # Interpolate RNG data
        if 'rng_timestamps' in combined_data and 'rng_values' in combined_data:
            rng_interp = self._interpolate_data(
                combined_data['rng_timestamps'],
                combined_data['rng_values'],
                time_grid
            )
            interpolated['rng_features'] = rng_interp
        
        # Interpolate EEG data
        eeg_features = []
        for key in combined_data.keys():
            if key.startswith('eeg_') and not key.endswith('_timestamps'):
                channel_data = self._interpolate_data(
                    combined_data['eeg_timestamps'],
                    combined_data[key],
                    time_grid
                )
                eeg_features.append(channel_data)
        
        if eeg_features:
            interpolated['eeg_features'] = np.column_stack(eeg_features)
        
        # Interpolate drawing data (as targets)
        if 'drawing_timestamps' in combined_data:
            # Colors as targets
            if 'drawing_colors_r' in combined_data:
                color_features = np.column_stack([
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_colors_r'], time_grid),
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_colors_g'], time_grid),
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_colors_b'], time_grid),
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_colors_a'], time_grid)
                ])
                interpolated['color_targets'] = color_features
            
            # Positions as targets
            if 'drawing_positions_x' in combined_data:
                position_features = np.column_stack([
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_positions_x'], time_grid),
                    self._interpolate_data(combined_data['drawing_timestamps'], combined_data['drawing_positions_y'], time_grid)
                ])
                interpolated['position_targets'] = position_features
            
            # Consciousness features
            if 'drawing_consciousness_layers' in combined_data:
                consciousness_features = self._interpolate_data(
                    combined_data['drawing_timestamps'], 
                    combined_data['drawing_consciousness_layers'], 
                    time_grid
                )
                interpolated['consciousness_targets'] = consciousness_features
            
            if 'drawing_pocket_dimensions' in combined_data:
                dimension_features = self._interpolate_data(
                    combined_data['drawing_timestamps'], 
                    combined_data['drawing_pocket_dimensions'], 
                    time_grid
                )
                interpolated['dimension_targets'] = dimension_features
        
        # Create sequences
        sequences = self._create_training_sequences(interpolated, sequence_length, prediction_horizon)
        
        return sequences
    
    def _interpolate_data(self, timestamps: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
        """
        Interpolate data onto target timestamps
        
        Args:
            timestamps: Original timestamps
            values: Original values
            target_times: Target timestamps
            
        Returns:
            Interpolated values
        """
        if len(values.shape) == 1:
            return np.interp(target_times, timestamps, values)
        else:
            # Multi-dimensional data
            interpolated = np.zeros((len(target_times), values.shape[1]))
            for i in range(values.shape[1]):
                interpolated[:, i] = np.interp(target_times, timestamps, values[:, i])
            return interpolated
    
    def _create_training_sequences(self, interpolated_data: Dict[str, np.ndarray],
                                 sequence_length: int, prediction_horizon: int) -> Dict[str, np.ndarray]:
        """
        Create training sequences from interpolated data
        
        Args:
            interpolated_data: Interpolated data
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon
            
        Returns:
            Training sequences
        """
        total_length = len(interpolated_data['timestamps'])
        num_sequences = max(0, total_length - sequence_length - prediction_horizon + 1)
        
        if num_sequences <= 0:
            self.logger.warning(f"Insufficient data for sequences: {total_length} < {sequence_length + prediction_horizon}")
            return {}
        
        sequences = {}
        
        # Create input sequences
        if 'rng_features' in interpolated_data:
            rng_features = interpolated_data['rng_features']
            rng_sequences = np.zeros((num_sequences, sequence_length, rng_features.shape[1]))
            for i in range(num_sequences):
                rng_sequences[i] = rng_features[i:i + sequence_length]
            sequences['rng_inputs'] = rng_sequences
        
        if 'eeg_features' in interpolated_data:
            eeg_features = interpolated_data['eeg_features']
            eeg_sequences = np.zeros((num_sequences, sequence_length, eeg_features.shape[1]))
            for i in range(num_sequences):
                eeg_sequences[i] = eeg_features[i:i + sequence_length]
            sequences['eeg_inputs'] = eeg_sequences
        
        # Combine RNG and EEG for mode 2
        if 'rng_inputs' in sequences and 'eeg_inputs' in sequences:
            combined_sequences = np.concatenate([sequences['rng_inputs'], sequences['eeg_inputs']], axis=2)
            sequences['combined_inputs'] = combined_sequences
        
        # Create target sequences
        for target_key in ['color_targets', 'position_targets', 'consciousness_targets', 'dimension_targets']:
            if target_key in interpolated_data:
                target_data = interpolated_data[target_key]
                if len(target_data.shape) == 1:
                    target_sequences = np.zeros((num_sequences,))
                    for i in range(num_sequences):
                        target_sequences[i] = target_data[i + sequence_length + prediction_horizon - 1]
                else:
                    target_sequences = np.zeros((num_sequences, target_data.shape[1]))
                    for i in range(num_sequences):
                        target_sequences[i] = target_data[i + sequence_length + prediction_horizon - 1]
                sequences[target_key] = target_sequences
        
        self.logger.info(f"Created {num_sequences} training sequences")
        for key, value in sequences.items():
            self.logger.info(f"  {key}: {value.shape}")
        
        return sequences


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = RealDataLoader("data")
    
    # Find and load sessions
    sessions = loader.load_multiple_sessions()
    
    if sessions:
        print(f"Loaded {len(sessions)} sessions")
        
        # Prepare training data
        training_data = loader.prepare_training_data(sessions, sequence_length=50)
        
        print("Training data prepared:")
        for key, value in training_data.items():
            print(f"  {key}: {value.shape}")
    else:
        print("No sessions loaded")