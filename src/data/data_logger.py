"""
Data Logging System

Comprehensive data logging for RNG, EEG, drawing actions, and timing
with configurable time offset compensation.
"""

import os
import sys
import time
import json
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, is_dataclass
import queue

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import numpy as np
# Ensure 'src' directory is on sys.path when running this module directly
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
except Exception:
    pass
# Prefer absolute imports; fall back to 'src.' when running this file directly
try:
    from hardware.truerng_v3 import RNGSample
    from hardware.emotiv_eeg import EEGSample
    from gui.painting_interface import DrawingAction
except ImportError:
    # Fallback if environment is unusual
    from src.hardware.truerng_v3 import RNGSample
    from src.hardware.emotiv_eeg import EEGSample
    from src.gui.painting_interface import DrawingAction


@dataclass
class SessionMetadata:
    """Metadata for a data collection session"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    participant_id: Optional[str]
    experiment_notes: str
    hardware_config: Dict[str, Any]
    drawing_delay_offset: float
    total_drawing_actions: int = 0
    total_rng_samples: int = 0
    total_eeg_samples: int = 0
    dial_visualization_enabled: bool = False


@dataclass
class SynchronizedDataPoint:
    """A single synchronized data point combining all streams"""
    timestamp: float
    corrected_timestamp: float  # Timestamp with delay compensation
    rng_data: Optional[List[float]]
    eeg_data: Optional[Dict[str, float]]
    drawing_action: Optional[Dict[str, Any]]
    dial_positions: Optional[Dict[int, Dict]]


class DataLogger:
    """Main data logging class"""
    
    def __init__(self, output_directory: str = "data", 
                 session_prefix: str = "session",
                 file_format: str = "hdf5",
                 compression: bool = True,
                 drawing_delay_offset: float = 0.2):
        """
        Initialize data logger
        
        Args:
            output_directory: Directory to save data files
            session_prefix: Prefix for session filenames
            file_format: File format ('hdf5', 'csv', 'json')
            compression: Whether to compress data files
            drawing_delay_offset: Time offset compensation in seconds
        """
        self.output_directory = output_directory
        self.session_prefix = session_prefix
        self.file_format = file_format
        self.compression = compression
        self.drawing_delay_offset = drawing_delay_offset
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Session state
        self.session_active = False
        self.session_metadata: Optional[SessionMetadata] = None
        self.session_start_time = 0.0
        
        # Data buffers
        self.rng_buffer: List[RNGSample] = []
        self.eeg_buffer: List[EEGSample] = []
        self.drawing_buffer: List[DrawingAction] = []
        self.dial_buffer: List[Tuple[float, Dict]] = []  # (timestamp, dial_positions)
        
        # Synchronization
        self.sync_data_buffer: List[SynchronizedDataPoint] = []
        self.buffer_lock = threading.Lock()
        
        # Background processing
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # File handles
        self.file_handles = {}
        
    def start_session(self, participant_id: Optional[str] = None,
                     experiment_notes: str = "",
                     hardware_config: Dict[str, Any] = None) -> str:
        """
        Start a new data logging session
        
        Args:
            participant_id: Optional participant identifier
            experiment_notes: Notes about the experiment
            hardware_config: Configuration of hardware devices
            
        Returns:
            Session ID
        """
        if self.session_active:
            self.stop_session()
            
        # Generate session ID
        session_id = f"{self.session_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create session metadata
        self.session_start_time = time.time()
        
        # Extract dial visualization flag from hardware config
        dial_viz_enabled = False
        if hardware_config and 'dial_visualization' in hardware_config:
            dial_viz_enabled = hardware_config['dial_visualization'] == 'Enabled'
        
        self.session_metadata = SessionMetadata(
            session_id=session_id,
            start_time=self.session_start_time,
            end_time=None,
            participant_id=participant_id,
            experiment_notes=experiment_notes,
            hardware_config=hardware_config or {},
            drawing_delay_offset=self.drawing_delay_offset,
            dial_visualization_enabled=dial_viz_enabled
        )
        
        # Clear buffers
        self.clear_buffers()
        
        # Start background processing
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self.processing_thread.start()
        
        # Initialize file handles
        self._initialize_files()
        
        self.session_active = True
        
        print(f"Started data logging session: {session_id}")
        return session_id
    def stop_session(self) -> Optional[str]:
        """
        Stop the current data logging session
        
        Returns:
            Path to the saved session file or None if no active session
        """
        if not self.session_active:
            return None
            
        # Stop background processing
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        # Update metadata counters
        if self.session_metadata:
            self.session_metadata.end_time = time.time()
            self.session_metadata.total_drawing_actions = len(self.drawing_buffer)
            self.session_metadata.total_rng_samples = len(self.rng_buffer)
            self.session_metadata.total_eeg_samples = len(self.eeg_buffer)
        
        # Save all data with error handling
        try:
            output_path = self._save_session_data()
        except Exception as e:
            print(f"Error saving session data: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save at least a minimal file
            session_id = self.session_metadata.session_id if self.session_metadata else "error_session"
            output_path = os.path.join(self.output_directory, f"{session_id}_error.json")
            try:
                error_data = {
                    'metadata': {
                        'session_id': session_id,
                        'error': str(e),
                        'timestamp': time.time(),
                        'rng_samples_count': len(self.rng_buffer),
                        'eeg_samples_count': len(self.eeg_buffer),
                        'drawing_actions_count': len(self.drawing_buffer)
                    },
                    'error_info': 'Data serialization failed, but session metadata saved'
                }
                with open(output_path, 'w') as f:
                    json.dump(error_data, f, indent=2)
                print(f"Saved error recovery file: {output_path}")
            except Exception as e2:
                print(f"Failed to save even error recovery file: {e2}")
                output_path = None
        
        # Close file handles
        self._close_files()
        
        # Clear session state
        self.session_active = False
        session_id = self.session_metadata.session_id if self.session_metadata else "unknown"
        self.session_metadata = None
        
        print(f"Stopped data logging session: {session_id}")
        print(f"Data saved to: {output_path}")
        
        return output_path
        
    def log_rng_sample(self, sample: RNGSample):
        """Log an RNG sample"""
        if not self.session_active:
            return
            
        with self.buffer_lock:
            self.rng_buffer.append(sample)
            
    def log_eeg_sample(self, sample: EEGSample):
        """Log an EEG sample"""
        if not self.session_active:
            return
            
        with self.buffer_lock:
            self.eeg_buffer.append(sample)
            
    def log_drawing_action(self, action: DrawingAction):
        """Log a drawing action with corrected timestamp"""
        if not self.session_active:
            return
            
        # Apply delay compensation
        action.timestamp -= self.drawing_delay_offset
        
        with self.buffer_lock:
            self.drawing_buffer.append(action)
            
    def log_dial_positions(self, positions: Dict[int, Dict]):
        """Log dial positions"""
        if not self.session_active:
            return
            
        timestamp = time.time()
        with self.buffer_lock:
            self.dial_buffer.append((timestamp, positions))
            
    def clear_buffers(self):
        """Clear all data buffers"""
        with self.buffer_lock:
            self.rng_buffer.clear()
            self.eeg_buffer.clear()
            self.drawing_buffer.clear()
            self.dial_buffer.clear()
            self.sync_data_buffer.clear()
            
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if not self.session_active or not self.session_metadata:
            return {}
            
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        with self.buffer_lock:
            return {
                'session_id': self.session_metadata.session_id,
                'duration': session_duration,
                'rng_samples': len(self.rng_buffer),
                'eeg_samples': len(self.eeg_buffer),
                'drawing_actions': len(self.drawing_buffer),
                'dial_updates': len(self.dial_buffer),
                'synchronized_points': len(self.sync_data_buffer)
            }
            
    def _processing_worker(self):
        """Background worker for data synchronization and processing"""
        while not self.stop_event.is_set():
            try:
                self._synchronize_data()
                time.sleep(0.1)  # Process every 100ms
            except Exception as e:
                print(f"Error in data processing worker: {e}")
                
    def _synchronize_data(self):
        """Synchronize data from different streams by timestamp"""
        if not self.session_active:
            return
            
        with self.buffer_lock:
            # Get latest samples from each buffer
            latest_samples = self._get_latest_samples_for_sync()
            
            for timestamp, samples in latest_samples.items():
                corrected_timestamp = timestamp - self.drawing_delay_offset
                
                sync_point = SynchronizedDataPoint(
                    timestamp=timestamp,
                    corrected_timestamp=corrected_timestamp,
                    rng_data=samples.get('rng'),
                    eeg_data=samples.get('eeg'),
                    drawing_action=samples.get('drawing'),
                    dial_positions=samples.get('dials')
                )
                
                self.sync_data_buffer.append(sync_point)
                
                # Limit buffer size
                if len(self.sync_data_buffer) > 10000:
                    self.sync_data_buffer = self.sync_data_buffer[-5000:]
                    
    def _get_latest_samples_for_sync(self) -> Dict[float, Dict[str, Any]]:
        """Get samples grouped by timestamp for synchronization"""
        # This is a simplified version - in practice, you'd want more
        # sophisticated temporal alignment
        
        samples_by_time = {}
        current_time = time.time()
        time_window = 0.1  # 100ms window for grouping
        
        # Process recent samples only
        recent_threshold = current_time - 1.0  # Last 1 second
        
        # Group RNG samples
        for sample in self.rng_buffer[-100:]:  # Last 100 samples
            if sample.timestamp > recent_threshold:
                time_key = round(sample.timestamp / time_window) * time_window
                if time_key not in samples_by_time:
                    samples_by_time[time_key] = {}
                samples_by_time[time_key]['rng'] = sample.normalized
                
        # Group EEG samples  
        for sample in self.eeg_buffer[-100:]:
            if sample.timestamp > recent_threshold:
                time_key = round(sample.timestamp / time_window) * time_window
                if time_key not in samples_by_time:
                    samples_by_time[time_key] = {}
                samples_by_time[time_key]['eeg'] = sample.channels
                
        # Group drawing actions
        for action in self.drawing_buffer[-50:]:
            if action.timestamp > recent_threshold:
                time_key = round(action.timestamp / time_window) * time_window
                if time_key not in samples_by_time:
                    samples_by_time[time_key] = {}
                samples_by_time[time_key]['drawing'] = self._action_to_dict(action)
                
        return samples_by_time

    def _action_to_dict(self, action: Any) -> Dict[str, Any]:
        """Safely convert a drawing action (dataclass or simple object) to a dict"""
        try:
            if is_dataclass(action):
                return asdict(action)
        except Exception:
            pass
        # Fallback: construct dict from known attributes
        result = {
            'timestamp': getattr(action, 'timestamp', 0.0),
            'action_type': getattr(action, 'action_type', ''),
            'position': getattr(action, 'position', (0, 0)),
            'color': getattr(action, 'color', (0, 0, 0, 255)),
            'brush_size': getattr(action, 'brush_size', 1),
            'pressure': getattr(action, 'pressure', 1.0),
        }
        # Optional fields
        result['consciousness_layer'] = getattr(action, 'consciousness_layer', 1)
        result['pocket_dimension'] = getattr(action, 'pocket_dimension', 1)
        meta = getattr(action, 'metadata', None)
        if meta is not None:
            result['metadata'] = meta
        return result
        
    def _initialize_files(self):
        """Initialize output files based on format"""
        if not self.session_metadata:
            return
            
        session_id = self.session_metadata.session_id
        
        if self.file_format == "hdf5" and HDF5_AVAILABLE:
            self._initialize_hdf5_file(session_id)
        elif self.file_format == "json":
            self._initialize_json_file(session_id)
        else:
            print(f"File format {self.file_format} not supported or dependencies missing")
            
    def _initialize_hdf5_file(self, session_id: str):
        """Initialize HDF5 file for efficient data storage"""
        filepath = os.path.join(self.output_directory, f"{session_id}.h5")
        
        compression_args = {'compression': 'gzip', 'compression_opts': 9} if self.compression else {}
        
        self.file_handles['hdf5'] = h5py.File(filepath, 'w')
        h5f = self.file_handles['hdf5']
        
        # Create groups
        h5f.create_group('metadata')
        h5f.create_group('rng_data')
        h5f.create_group('eeg_data')
        h5f.create_group('drawing_data')
        h5f.create_group('dial_data')
        h5f.create_group('synchronized_data')
        
        # Save metadata
        meta_group = h5f['metadata']
        if self.session_metadata:
            meta_group.attrs['session_id'] = self.session_metadata.session_id
            meta_group.attrs['start_time'] = self.session_metadata.start_time
            meta_group.attrs['drawing_delay_offset'] = self.session_metadata.drawing_delay_offset
            meta_group.attrs['dial_visualization_enabled'] = self.session_metadata.dial_visualization_enabled
            
    def _initialize_json_file(self, session_id: str):
        """Initialize JSON file structure"""
        filepath = os.path.join(self.output_directory, f"{session_id}.json")
        self.file_handles['json_path'] = filepath
        
        # Create initial structure
        initial_data = {
            'metadata': asdict(self.session_metadata) if self.session_metadata else {},
            'rng_data': [],
            'eeg_data': [],
            'drawing_data': [],
            'dial_data': [],
            'synchronized_data': []
        }
        
        with open(filepath, 'w') as f:
            json.dump(initial_data, f, indent=2)
            
    def _save_session_data(self) -> str:
        """Save all session data to files"""
        if not self.session_metadata:
            return ""
            
        session_id = self.session_metadata.session_id
        
        if self.file_format == "hdf5" and HDF5_AVAILABLE:
            return self._save_hdf5_data(session_id)
        elif self.file_format == "json":
            return self._save_json_data(session_id)
        elif self.file_format == "csv" and PANDAS_AVAILABLE:
            return self._save_csv_data(session_id)
        else:
            return self._save_fallback_data(session_id)
            
    def _save_hdf5_data(self, session_id: str) -> str:
        """Save data in HDF5 format"""
        filepath = os.path.join(self.output_directory, f"{session_id}.h5")
        
        if 'hdf5' not in self.file_handles:
            return filepath
            
        h5f = self.file_handles['hdf5']
        
        # Save RNG data
        if self.rng_buffer:
            rng_timestamps = [s.timestamp for s in self.rng_buffer]
            rng_values = [s.normalized for s in self.rng_buffer]
            
            h5f['rng_data'].create_dataset('timestamps', data=rng_timestamps)
            h5f['rng_data'].create_dataset('values', data=rng_values)
            
        # Save EEG data
        if self.eeg_buffer:
            eeg_timestamps = [s.timestamp for s in self.eeg_buffer]
            h5f['eeg_data'].create_dataset('timestamps', data=eeg_timestamps)
            
            # Save each channel separately
            if self.eeg_buffer:
                for channel in self.eeg_buffer[0].channels.keys():
                    channel_data = [s.channels.get(channel, 0) for s in self.eeg_buffer]
                    h5f['eeg_data'].create_dataset(f'channel_{channel}', data=channel_data)
                    
        # Save drawing data
        if self.drawing_buffer:
            # Save each field separately to handle different data types properly
            timestamps = [action.timestamp for action in self.drawing_buffer]
            action_types = [str(action.action_type).encode('ascii', errors='replace') for action in self.drawing_buffer]
            positions_x = [action.position[0] for action in self.drawing_buffer]
            positions_y = [action.position[1] for action in self.drawing_buffer]
            colors_r = [action.color[0] for action in self.drawing_buffer]
            colors_g = [action.color[1] for action in self.drawing_buffer]
            colors_b = [action.color[2] for action in self.drawing_buffer]
            colors_a = [action.color[3] for action in self.drawing_buffer]
            brush_sizes = [action.brush_size for action in self.drawing_buffer]
            pressures = [action.pressure for action in self.drawing_buffer]
            
            # Save as separate datasets
            h5f['drawing_data'].create_dataset('timestamps', data=timestamps)
            h5f['drawing_data'].create_dataset('action_types', data=action_types)
            h5f['drawing_data'].create_dataset('positions_x', data=positions_x)
            h5f['drawing_data'].create_dataset('positions_y', data=positions_y)
            h5f['drawing_data'].create_dataset('colors_r', data=colors_r)
            h5f['drawing_data'].create_dataset('colors_g', data=colors_g)
            h5f['drawing_data'].create_dataset('colors_b', data=colors_b)
            h5f['drawing_data'].create_dataset('colors_a', data=colors_a)
            h5f['drawing_data'].create_dataset('brush_sizes', data=brush_sizes)
            h5f['drawing_data'].create_dataset('pressures', data=pressures)
            
            # Also save enhanced fields if they exist
            if hasattr(self.drawing_buffer[0], 'consciousness_layer'):
                consciousness_layers = [getattr(action, 'consciousness_layer', 1) for action in self.drawing_buffer]
                h5f['drawing_data'].create_dataset('consciousness_layers', data=consciousness_layers)
                
            if hasattr(self.drawing_buffer[0], 'pocket_dimension'):
                pocket_dimensions = [getattr(action, 'pocket_dimension', 1) for action in self.drawing_buffer]
                h5f['drawing_data'].create_dataset('pocket_dimensions', data=pocket_dimensions)
            
            # Save per-action metadata as JSON bytes (variable-length)
            metadata_strings = []
            for action in self.drawing_buffer:
                if hasattr(action, 'metadata') and getattr(action, 'metadata', None):
                    try:
                        metadata_strings.append(json.dumps(action.metadata, default=str).encode('utf-8', errors='replace'))
                    except Exception:
                        metadata_strings.append(b'{}')
                else:
                    metadata_strings.append(b'{}')
            try:
                vlen_bytes = h5py.special_dtype(vlen=bytes) if hasattr(h5py, 'special_dtype') else None
                if vlen_bytes is not None:
                    h5f['drawing_data'].create_dataset('metadata', data=metadata_strings, dtype=vlen_bytes)
                else:
                    # Fallback without explicit dtype
                    h5f['drawing_data'].create_dataset('metadata', data=metadata_strings)
            except Exception as e:
                print(f"Warning: failed to save drawing metadata dataset: {e}")
        
        # Save dial data
        if self.dial_buffer:
            dial_timestamps = [t for t, _ in self.dial_buffer]
            # Convert dial positions to JSON strings for safe storage
            dial_position_strings = []
            for _, positions in self.dial_buffer:
                try:
                    json_str = json.dumps(positions, default=str).encode('utf-8', errors='replace')
                except:
                    json_str = b'{}'
                dial_position_strings.append(json_str)
            
            h5f['dial_data'].create_dataset('timestamps', data=dial_timestamps)
            try:
                vlen_bytes = h5py.special_dtype(vlen=bytes) if hasattr(h5py, 'special_dtype') else None
                if vlen_bytes is not None:
                    h5f['dial_data'].create_dataset('positions', data=dial_position_strings, dtype=vlen_bytes)
                else:
                    h5f['dial_data'].create_dataset('positions', data=dial_position_strings)
            except Exception as e:
                print(f"Warning: failed to save dial positions dataset: {e}")

        # Update metadata attributes with proper string handling
        meta_group = h5f['metadata']
        if self.session_metadata:
            for key, value in asdict(self.session_metadata).items():
                if value is None:
                    continue
                if isinstance(value, str):
                    meta_group.attrs[key] = value.encode('utf-8', errors='replace')
                elif isinstance(value, dict):
                    json_str = json.dumps(value, default=str)
                    meta_group.attrs[key] = json_str.encode('utf-8', errors='replace')
                else:
                    meta_group.attrs[key] = value

        return filepath
        
    def _safe_json_serialize(self, obj):
        """Safely serialize objects to JSON-compatible format"""
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False
        
        if has_numpy:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
        
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif hasattr(obj, 'tolist'):  # Duck typing for array-like objects
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)

    def _save_json_data(self, session_id: str) -> str:
        """Save data in JSON format"""
        filepath = os.path.join(self.output_directory, f"{session_id}.json")
        
        try:
            data = {
                'metadata': asdict(self.session_metadata) if self.session_metadata else {},
                'rng_data': [asdict(s) for s in self.rng_buffer],
                'eeg_data': [asdict(s) for s in self.eeg_buffer],
                'drawing_data': [asdict(a) for a in self.drawing_buffer],
                'dial_data': [{'timestamp': t, 'positions': p} for t, p in self.dial_buffer],
                'synchronized_data': [asdict(s) for s in self.sync_data_buffer]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=self._safe_json_serialize)
                
        except Exception as e:
            # Fallback: try to save with more aggressive serialization
            print(f"Warning: JSON serialization failed with error: {e}")
            print("Attempting fallback serialization...")
            
            try:
                data = {
                    'metadata': self._serialize_metadata(),
                    'rng_data': self._serialize_rng_data(),
                    'eeg_data': self._serialize_eeg_data(),
                    'drawing_data': self._serialize_drawing_data(),
                    'dial_data': [{'timestamp': float(t), 'positions': dict(p)} for t, p in self.dial_buffer],
                    'synchronized_data': []  # Skip sync data if problematic
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=self._safe_json_serialize)
                    
            except Exception as e2:
                print(f"Error: Both JSON serialization attempts failed: {e2}")
                # Create minimal file with just metadata
                minimal_data = {
                    'metadata': {
                        'session_id': session_id,
                        'error': f'Serialization failed: {str(e2)}',
                        'timestamp': time.time()
                    },
                    'rng_data': [],
                    'eeg_data': [],
                    'drawing_data': [],
                    'dial_data': [],
                    'synchronized_data': []
                }
                with open(filepath, 'w') as f:
                    json.dump(minimal_data, f, indent=2)
            
        return filepath
    
    def _serialize_metadata(self):
        """Safely serialize metadata"""
        if not self.session_metadata:
            return {}
        
        meta_dict = asdict(self.session_metadata)
        # Clean up any problematic data types
        for key, value in meta_dict.items():
            if hasattr(value, 'tolist'):  # NumPy array
                meta_dict[key] = value.tolist()
            elif isinstance(value, bytes):
                meta_dict[key] = value.decode('utf-8', errors='replace')
        return meta_dict
    
    def _serialize_rng_data(self):
        """Safely serialize RNG data"""
        serialized = []
        for sample in self.rng_buffer:
            try:
                sample_dict = asdict(sample)
                # Ensure raw_bytes is properly handled
                if isinstance(sample_dict.get('raw_bytes'), bytes):
                    sample_dict['raw_bytes'] = sample_dict['raw_bytes'].decode('latin-1', errors='replace')
                # Ensure lists are properly converted
                if hasattr(sample_dict.get('values'), 'tolist'):
                    sample_dict['values'] = sample_dict['values'].tolist()
                if hasattr(sample_dict.get('normalized'), 'tolist'):
                    sample_dict['normalized'] = sample_dict['normalized'].tolist()
                serialized.append(sample_dict)
            except Exception as e:
                print(f"Warning: Skipping problematic RNG sample: {e}")
        return serialized
    
    def _serialize_eeg_data(self):
        """Safely serialize EEG data"""
        serialized = []
        for sample in self.eeg_buffer:
            try:
                sample_dict = asdict(sample)
                # Clean up channel data
                if 'channels' in sample_dict:
                    channels = sample_dict['channels']
                    for channel, value in channels.items():
                        if hasattr(value, 'item'):  # NumPy scalar
                            channels[channel] = float(value.item())
                serialized.append(sample_dict)
            except Exception as e:
                print(f"Warning: Skipping problematic EEG sample: {e}")
        return serialized
    
    def _serialize_drawing_data(self):
        """Safely serialize drawing data"""
        serialized = []
        for action in self.drawing_buffer:
            try:
                action_dict = asdict(action)
                # Ensure position is a simple tuple/list
                if hasattr(action_dict.get('position'), 'tolist'):
                    action_dict['position'] = action_dict['position'].tolist()
                # Ensure color is a simple tuple/list
                if hasattr(action_dict.get('color'), 'tolist'):
                    action_dict['color'] = action_dict['color'].tolist()
                serialized.append(action_dict)
            except Exception as e:
                print(f"Warning: Skipping problematic drawing action: {e}")
        return serialized
        
    def _save_csv_data(self, session_id: str) -> str:
        """Save data in CSV format using pandas"""
        output_dir = os.path.join(self.output_directory, session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        meta_file = os.path.join(output_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(asdict(self.session_metadata) if self.session_metadata else {}, f, indent=2, default=str)
            
        # Save RNG data
        if self.rng_buffer:
            rng_df = pd.DataFrame([{
                'timestamp': s.timestamp,
                **{f'value_{i}': v for i, v in enumerate(s.normalized)}
            } for s in self.rng_buffer])
            rng_df.to_csv(os.path.join(output_dir, 'rng_data.csv'), index=False)
            
        # Save EEG data
        if self.eeg_buffer:
            eeg_df = pd.DataFrame([{
                'timestamp': s.timestamp,
                **s.channels
            } for s in self.eeg_buffer])
            eeg_df.to_csv(os.path.join(output_dir, 'eeg_data.csv'), index=False)
            
        # Save drawing data
        if self.drawing_buffer:
            drawing_df = pd.DataFrame([{
                'timestamp': a.timestamp,
                'action_type': a.action_type,
                'x': a.position[0],
                'y': a.position[1],
                'r': a.color[0],
                'g': a.color[1],
                'b': a.color[2],
                'a': a.color[3],
                'brush_size': a.brush_size,
                'pressure': a.pressure
            } for a in self.drawing_buffer])
            drawing_df.to_csv(os.path.join(output_dir, 'drawing_data.csv'), index=False)
            
        return output_dir
        
    def _save_fallback_data(self, session_id: str) -> str:
        """Fallback save method using basic JSON"""
        return self._save_json_data(session_id)
        
    def _close_files(self):
        """Close all open file handles"""
        for handle in self.file_handles.values():
            try:
                if hasattr(handle, 'close'):
                    handle.close()
            except Exception as e:
                print(f"Error closing file handle: {e}")
                
        self.file_handles.clear()


# Data analysis utilities
class DataAnalyzer:
    """Utilities for analyzing logged consciousness data"""
    
    @staticmethod
    def load_session_data(filepath: str) -> Dict[str, Any]:
        """Load session data from file"""
        if filepath.endswith('.h5') and HDF5_AVAILABLE:
            return DataAnalyzer._load_hdf5_data(filepath)
        elif filepath.endswith('.json'):
            return DataAnalyzer._load_json_data(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
    @staticmethod
    def _load_hdf5_data(filepath: str) -> Dict[str, Any]:
        """Load HDF5 data"""
        data = {}
        with h5py.File(filepath, 'r') as h5f:
            # Load metadata
            data['metadata'] = dict(h5f['metadata'].attrs)
            
            # Load datasets
            for group_name in ['rng_data', 'eeg_data', 'drawing_data', 'dial_data']:
                if group_name in h5f:
                    data[group_name] = {}
                    for dataset_name in h5f[group_name].keys():
                        data[group_name][dataset_name] = h5f[group_name][dataset_name][:]
                        
        return data
        
    @staticmethod
    def _load_json_data(filepath: str) -> Dict[str, Any]:
        """Load JSON data"""
        with open(filepath, 'r') as f:
            return json.load(f)
            
    @staticmethod
    def compute_synchronization_quality(data: Dict[str, Any]) -> Dict[str, float]:
        """Compute quality metrics for data synchronization"""
        # This would analyze timing alignment between streams
        # For now, return placeholder metrics
        return {
            'temporal_alignment_score': 0.85,
            'data_completeness': 0.92,
            'sampling_rate_consistency': 0.88
        }


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = DataLogger(
        output_directory="test_data",
        drawing_delay_offset=0.15
    )
    
    # Start session
    session_id = logger.start_session(
        participant_id="test_001",
        experiment_notes="Testing data logging system",
        hardware_config={'rng_device': 'TrueRNG_V3', 'eeg_device': 'Emotiv_EPOC'}
    )
    
    # Simulate some data
    import random
    for i in range(10):
        # Mock RNG sample
        mock_rng = type('MockRNG', (), {
            'timestamp': time.time(),
            'normalized': [random.random() for _ in range(8)]
        })()
        logger.log_rng_sample(mock_rng)
        
        # Mock drawing action
        mock_action = type('MockAction', (), {
            'timestamp': time.time(),
            'action_type': 'stroke_continue',
            'position': (random.randint(0, 800), random.randint(0, 600)),
            'color': (255, 0, 0, 255),
            'brush_size': 10,
            'pressure': 1.0
        })()
        logger.log_drawing_action(mock_action)
        
        time.sleep(0.1)
        
    # Get stats
    stats = logger.get_session_stats()
    print(f"Session stats: {stats}")
    
    # Stop session
    output_path = logger.stop_session()
    print(f"Data saved to: {output_path}")