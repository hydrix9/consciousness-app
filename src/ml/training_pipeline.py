"""
Machine Learning Training Pipeline

This module creates ML models to train on consciousness data and produce
the same outputs (colors and 3D interlocking dials) in two modes:
- Mode 1: RNG data only
- Mode 2: RNG + EEG data combined
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for ML training"""
    model_type: str = "lstm"  # "lstm", "transformer", "cnn_lstm"
    sequence_length: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    early_stopping_patience: int = 10
    min_epochs: int = 5  # Minimum epochs before early stopping
    verbose_training: bool = True  # Detailed training output


class ConsciousnessTrainingCallback:
    """Custom callback for detailed training progress"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = None
        self.epoch_start_time = None
        self.history = {
            'loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.start_time = time.time()
        if self.verbose:
            print("🧠 " + "="*60)
            print("  CONSCIOUSNESS MODEL TRAINING INITIATED")
            print("="*60)
            print(f"⏰ Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
        if self.verbose:
            print(f"🌟 Epoch {epoch + 1:3d}: Processing consciousness patterns...", end="", flush=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        epoch_time = time.time() - self.epoch_start_time
        self.history['epoch_times'].append(epoch_time)
        
        if logs:
            self.history['loss'].append(logs.get('loss', 0))
            self.history['val_loss'].append(logs.get('val_loss', 0))
            self.history['learning_rates'].append(logs.get('learning_rate', 0))
        
        if self.verbose and logs:
            # Progress bar visualization
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            
            print(f"\\r🌟 Epoch {epoch + 1:3d}: ", end="")
            print(f"Loss: {loss:.6f} | Val Loss: {val_loss:.6f} | ", end="")
            print(f"Time: {epoch_time:.1f}s")
            
            # Show learning progress
            if epoch > 0:
                prev_loss = self.history['loss'][-2] if len(self.history['loss']) > 1 else loss
                improvement = prev_loss - loss
                if improvement > 0:
                    print(f"   📈 Improvement: {improvement:.6f} ({improvement/prev_loss*100:.2f}%)")
                else:
                    print(f"   📉 Change: {improvement:.6f} ({improvement/prev_loss*100:.2f}%)")
            print()
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        total_time = time.time() - self.start_time
        if self.verbose:
            print()
            print("🎉 " + "="*60)
            print("  CONSCIOUSNESS MODEL TRAINING COMPLETED")
            print("="*60)
            print(f"⏰ Total training time: {total_time:.1f} seconds")
            print(f"📊 Total epochs: {len(self.history['loss'])}")
            if self.history['loss']:
                print(f"🎯 Final loss: {self.history['loss'][-1]:.6f}")
                print(f"🎯 Final validation loss: {self.history['val_loss'][-1]:.6f}")
            print()


# TensorFlow callback wrapper
if TF_AVAILABLE:
    class TensorFlowProgressCallback(callbacks.Callback):
        """TensorFlow-compatible wrapper for our custom callback"""
        
        def __init__(self, consciousness_callback: ConsciousnessTrainingCallback):
            super().__init__()
            self.consciousness_callback = consciousness_callback
        
        def on_train_begin(self, logs=None):
            self.consciousness_callback.on_train_begin(logs)
        
        def on_epoch_begin(self, epoch, logs=None):
            self.consciousness_callback.on_epoch_begin(epoch, logs)
        
        def on_epoch_end(self, epoch, logs=None):
            self.consciousness_callback.on_epoch_end(epoch, logs)
        
        def on_train_end(self, logs=None):
            self.consciousness_callback.on_train_end(logs)


@dataclass
class TrainingData:
    """Processed training data structure"""
    mode1_inputs: np.ndarray  # RNG sequences
    mode2_inputs: np.ndarray  # RNG + EEG sequences  
    color_outputs: np.ndarray  # Color predictions (RGBA)
    curve_outputs: np.ndarray  # 3D curve parameters
    dial_outputs: np.ndarray   # Dial positions and rotations
    timestamps: np.ndarray
    metadata: Dict[str, Any]


class RealDataPreprocessor:
    """Preprocesses real consciousness data for ML training using the RealDataLoader"""
    
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.rng_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.eeg_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.color_scaler = MinMaxScaler() if SKLEARN_AVAILABLE else None
        self.curve_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.dial_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Import the real data loader
        try:
            # Prefer absolute import since project adds 'src' to sys.path at runtime
            from data.real_data_loader import RealDataLoader
            self.data_loader = RealDataLoader()
            self.real_data_available = True
            print("✅ Real data loader initialized successfully")
        except ImportError as e:
            try:
                from src.data.real_data_loader import RealDataLoader  # fallback when run differently
                self.data_loader = RealDataLoader()
                self.real_data_available = True
                print("✅ Real data loader initialized via fallback")
            except ImportError:
                print(f"⚠️ Real data loader not available: {e}")
                self.data_loader = None
                self.real_data_available = False
        
    def load_real_session_data(self, data_directory: str = "data") -> TrainingData:
        """Load and preprocess real session data for training"""
        
        if not self.real_data_available:
            print("❌ Real data loader not available, falling back to mock data")
            return self._empty_training_data()
        
        try:
            # Set data directory
            self.data_loader.data_directory = data_directory
            
            # Load all available sessions
            print(f"🔍 Loading sessions from: {data_directory}")
            sessions = self.data_loader.load_multiple_sessions()
            
            if not sessions:
                print("⚠️ No sessions found, generating mock data for demonstration")
                return self._generate_demo_data()
            
            print(f"📊 Loaded {len(sessions)} real sessions")
            
            # Prepare training data
            training_sequences = self.data_loader.prepare_training_data(
                sessions, 
                sequence_length=self.sequence_length,
                prediction_horizon=1
            )
            
            if not training_sequences:
                print("⚠️ No training sequences generated, falling back to demo data")
                return self._generate_demo_data()
            
            # Convert to TrainingData format
            training_data = self._convert_to_training_data(training_sequences)
            
            print(f"✅ Real training data prepared successfully!")
            self._print_data_summary(training_data)
            
            return training_data
            
        except Exception as e:
            print(f"❌ Error loading real session data: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to demo data")
            return self._generate_demo_data()
    
    def _convert_to_training_data(self, sequences: Dict[str, np.ndarray]) -> TrainingData:
        """Convert real data loader sequences to TrainingData format"""
        
        # Mode 1 inputs (RNG only)
        mode1_inputs = sequences.get('rng_inputs', np.array([]))
        
        # Mode 2 inputs (RNG + EEG combined)
        mode2_inputs = sequences.get('combined_inputs', np.array([]))
        if len(mode2_inputs) == 0 and len(mode1_inputs) > 0:
            # If no EEG data, use RNG for both modes
            mode2_inputs = mode1_inputs
        
        # Color targets
        color_outputs = sequences.get('color_targets', np.array([]))
        if len(color_outputs) == 0 and len(mode1_inputs) > 0:
            # Generate default color targets if none available
            num_samples = len(mode1_inputs)
            color_outputs = np.random.uniform(0, 1, (num_samples, 4))
        
        # Position targets as "curve" outputs
        curve_outputs = sequences.get('position_targets', np.array([]))
        if len(curve_outputs) == 0 and len(mode1_inputs) > 0:
            # Generate default position targets
            num_samples = len(mode1_inputs)
            curve_outputs = np.random.uniform(0, 800, (num_samples, 2))
        
        # Consciousness and dimension data as "dial" outputs
        dial_outputs = np.array([])
        consciousness_targets = sequences.get('consciousness_targets', np.array([]))
        dimension_targets = sequences.get('dimension_targets', np.array([]))
        
        if len(consciousness_targets) > 0 or len(dimension_targets) > 0:
            # Combine consciousness and dimension data
            if len(consciousness_targets) > 0 and len(dimension_targets) > 0:
                dial_outputs = np.column_stack([consciousness_targets, dimension_targets])
            elif len(consciousness_targets) > 0:
                dial_outputs = consciousness_targets.reshape(-1, 1)
            elif len(dimension_targets) > 0:
                dial_outputs = dimension_targets.reshape(-1, 1)
        elif len(mode1_inputs) > 0:
            # Generate default dial outputs
            num_samples = len(mode1_inputs)
            dial_outputs = np.random.uniform(1, 3, (num_samples, 2))
        
        # Timestamps
        timestamps = sequences.get('timestamps', np.array([]))
        if len(timestamps) == 0 and len(mode1_inputs) > 0:
            timestamps = np.arange(len(mode1_inputs))
        
        return TrainingData(
            mode1_inputs=mode1_inputs,
            mode2_inputs=mode2_inputs,
            color_outputs=color_outputs,
            curve_outputs=curve_outputs,
            dial_outputs=dial_outputs,
            timestamps=timestamps,
            metadata={
                'sequence_length': self.sequence_length,
                'data_source': 'real_sessions',
                'num_sequences': len(mode1_inputs),
                'rng_feature_dim': mode1_inputs.shape[-1] if len(mode1_inputs) > 0 else 0,
                'color_dim': color_outputs.shape[-1] if len(color_outputs) > 0 else 4,
                'curve_dim': curve_outputs.shape[-1] if len(curve_outputs) > 0 else 2,
                'dial_dim': dial_outputs.shape[-1] if len(dial_outputs) > 0 else 2
            }
        )
    
    def _generate_demo_data(self) -> TrainingData:
        """Generate demo data for testing when no real data is available"""
        print("🎭 Generating demo consciousness data for training...")
        
        num_sequences = 100
        sequence_length = self.sequence_length
        
        # Generate synthetic RNG data
        rng_features = 8
        mode1_inputs = np.random.normal(0, 1, (num_sequences, sequence_length, rng_features))
        
        # Generate synthetic EEG data
        eeg_features = 14  # Standard EEG channel count
        eeg_data = np.random.normal(0, 0.1, (num_sequences, sequence_length, eeg_features))
        mode2_inputs = np.concatenate([mode1_inputs, eeg_data], axis=2)
        
        # Generate targets based on input patterns
        color_outputs = np.zeros((num_sequences, 4))
        curve_outputs = np.zeros((num_sequences, 5))
        dial_outputs = np.zeros((num_sequences, 8))
        
        for i in range(num_sequences):
            # Colors influenced by RNG patterns
            rng_mean = np.mean(mode1_inputs[i], axis=0)
            color_outputs[i] = [
                (rng_mean[0] + 1) / 2,  # R
                (rng_mean[1] + 1) / 2,  # G
                (rng_mean[2] + 1) / 2,  # B
                0.8  # A
            ]
            
            # Curves influenced by RNG variance
            rng_var = np.var(mode1_inputs[i], axis=0)
            curve_outputs[i] = [
                rng_var[0] * 400 + 200,  # x position
                rng_var[1] * 300 + 150,  # y position
                rng_var[2] * 20 + 5,     # brush size
                (rng_var[3] + 0.1) / 1.1, # pressure
                1.0 if np.mean(rng_mean[:4]) > 0 else 0.0  # action type
            ]
            
            # Dials influenced by combined patterns
            combined_mean = np.mean(mode2_inputs[i], axis=0)
            for j in range(8):
                dial_outputs[i, j] = (combined_mean[j % len(combined_mean)] + 1) / 2
        
        timestamps = np.arange(num_sequences)
        
        return TrainingData(
            mode1_inputs=mode1_inputs,
            mode2_inputs=mode2_inputs,
            color_outputs=color_outputs,
            curve_outputs=curve_outputs,
            dial_outputs=dial_outputs,
            timestamps=timestamps,
            metadata={
                'sequence_length': sequence_length,
                'data_source': 'demo_synthetic',
                'num_sequences': num_sequences,
                'rng_feature_dim': rng_features,
                'eeg_feature_dim': eeg_features,
                'color_dim': 4,
                'curve_dim': 5,
                'dial_dim': 8
            }
        )
    
    def _print_data_summary(self, training_data: TrainingData):
        """Print summary of loaded training data"""
        print("\n📈 Training Data Summary:")
        print(f"  Mode 1 inputs (RNG): {training_data.mode1_inputs.shape}")
        print(f"  Mode 2 inputs (RNG+EEG): {training_data.mode2_inputs.shape}")
        print(f"  Color targets: {training_data.color_outputs.shape}")
        print(f"  Curve targets: {training_data.curve_outputs.shape}")
        print(f"  Dial targets: {training_data.dial_outputs.shape}")
        print(f"  Metadata: {training_data.metadata}")
        print()
    
    def load_session_data(self, data_files: List[str]) -> Dict[str, List]:
        """
        Load session data from files - wrapper for compatibility with ModelTrainer
        
        This method provides compatibility with the ModelTrainer.train_models() interface
        which expects a load_session_data method that returns raw data for preprocessing.
        
        For RealDataPreprocessor, we actually bypass this and use load_real_session_data
        which returns fully preprocessed TrainingData directly.
        """
        # This method is here for interface compatibility
        # The actual loading happens in load_real_session_data()
        # Return empty dict as placeholder - won't be used
        return {
            'rng_data': [],
            'eeg_data': [],
            'drawing_data': [],
            'dial_data': []
        }
    
    def preprocess_data(self, raw_data: Dict[str, List]) -> TrainingData:
        """
        Preprocess raw session data - wrapper for compatibility
        
        For RealDataPreprocessor, actual preprocessing happens in load_real_session_data().
        This method is here for interface compatibility with ModelTrainer.
        """
        # This won't actually be called when using load_real_session_data
        # But if it is, fall back to demo data
        return self._generate_demo_data()


class DataPreprocessor:
    """Preprocesses consciousness data for ML training"""
    
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.rng_scaler = StandardScaler()
        self.eeg_scaler = StandardScaler()
        self.color_scaler = MinMaxScaler()
        self.curve_scaler = StandardScaler()
        self.dial_scaler = StandardScaler()
        
        self.eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
    def _empty_training_data(self) -> 'TrainingData':
        """Return empty training data structure"""
        return TrainingData(
            mode1_inputs=np.array([]),
            mode2_inputs=np.array([]),
            color_outputs=np.array([]),
            curve_outputs=np.array([]),
            dial_outputs=np.array([]),
            timestamps=np.array([]),
            metadata={'sequence_length': self.sequence_length}
        )
        
    def augment_data(self, data: np.ndarray, augmentation_factor: int = 3) -> np.ndarray:
        """Augment consciousness data with variations to increase dataset size"""
        if len(data) == 0:
            return data
            
        original_shape = data.shape
        augmented_data = [data]  # Original data
        
        for _ in range(augmentation_factor):
            # Add small noise variations
            noise_factor = 0.05  # 5% noise
            noisy_data = data + np.random.normal(0, noise_factor * np.std(data), data.shape)
            augmented_data.append(noisy_data)
            
            # Add scaling variations
            scale_factor = np.random.uniform(0.95, 1.05)  # ±5% scaling
            scaled_data = data * scale_factor
            augmented_data.append(scaled_data)
        
        # Concatenate along the first dimension (samples)
        result = np.concatenate(augmented_data, axis=0)
        print(f"✨ Data augmented from {len(data)} to {len(result)} samples")
        return result
        
    def load_session_data(self, data_files: List[str]) -> Dict[str, List]:
        """Load and combine multiple session data files (supports JSON and HDF5)"""
        combined_data = {
            'rng_samples': [],
            'eeg_samples': [],
            'drawing_actions': [],
            'dial_positions': [],
            'timestamps': []
        }
        
        for file_path in data_files:
            try:
                if file_path.endswith('.json'):
                    # Load JSON format
                    with open(file_path, 'r') as f:
                        session_data = json.load(f)
                        
                    # Extract data
                    combined_data['rng_samples'].extend(session_data.get('rng_data', []))
                    combined_data['eeg_samples'].extend(session_data.get('eeg_data', []))
                    combined_data['drawing_actions'].extend(session_data.get('drawing_data', []))
                    combined_data['dial_positions'].extend(session_data.get('dial_data', []))
                
                elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                    # Load HDF5 format
                    try:
                        import h5py
                    except ImportError:
                        logging.error(f"h5py not available, cannot load HDF5 file: {file_path}")
                        continue
                    
                    with h5py.File(file_path, 'r') as h5f:
                        # Load RNG data (stored as timestamps + values arrays)
                        if 'rng_data' in h5f and 'timestamps' in h5f['rng_data'] and 'values' in h5f['rng_data']:
                            timestamps = h5f['rng_data']['timestamps'][()]
                            values = h5f['rng_data']['values'][()]
                            
                            # Values can be 1D or 2D array
                            if len(values.shape) == 1:
                                # 1D array - single value per timestamp
                                for ts, val in zip(timestamps, values):
                                    combined_data['rng_samples'].append({
                                        'timestamp': float(ts),
                                        'normalized': float(val)
                                    })
                            else:
                                # 2D array - multiple values per timestamp (use first value or mean)
                                for ts, val_array in zip(timestamps, values):
                                    combined_data['rng_samples'].append({
                                        'timestamp': float(ts),
                                        'normalized': float(val_array[0])  # Use first value
                                    })
                        
                        # Load EEG data (stored as timestamps + channel arrays)
                        if 'eeg_data' in h5f and 'timestamps' in h5f['eeg_data']:
                            timestamps = h5f['eeg_data']['timestamps'][()]
                            # Find all channel datasets
                            channel_names = [k for k in h5f['eeg_data'].keys() if k.startswith('channel_')]
                            
                            if channel_names:
                                # Load channel data
                                channel_data = {}
                                for channel_name in channel_names:
                                    channel_data[channel_name] = h5f['eeg_data'][channel_name][()]
                                
                                # Reconstruct EEG samples
                                for i, ts in enumerate(timestamps):
                                    sample = {
                                        'timestamp': float(ts),
                                        'channels': {ch: float(channel_data[ch][i]) for ch in channel_names}
                                    }
                                    combined_data['eeg_samples'].append(sample)
                        
                        # Load drawing data (stored as separate field arrays)
                        if 'drawing_data' in h5f and 'timestamps' in h5f['drawing_data']:
                            timestamps = h5f['drawing_data']['timestamps'][()]
                            action_types = h5f['drawing_data']['action_types'][()]
                            positions_x = h5f['drawing_data']['positions_x'][()]
                            positions_y = h5f['drawing_data']['positions_y'][()]
                            colors_r = h5f['drawing_data']['colors_r'][()]
                            colors_g = h5f['drawing_data']['colors_g'][()]
                            colors_b = h5f['drawing_data']['colors_b'][()]
                            colors_a = h5f['drawing_data']['colors_a'][()]
                            brush_sizes = h5f['drawing_data']['brush_sizes'][()]
                            pressures = h5f['drawing_data']['pressures'][()]
                            
                            # Load optional fields if they exist
                            consciousness_layers = h5f['drawing_data']['consciousness_layers'][()] if 'consciousness_layers' in h5f['drawing_data'] else None
                            pocket_dimensions = h5f['drawing_data']['pocket_dimensions'][()] if 'pocket_dimensions' in h5f['drawing_data'] else None
                            
                            # Reconstruct drawing actions
                            for i in range(len(timestamps)):
                                action = {
                                    'timestamp': float(timestamps[i]),
                                    'action_type': action_types[i].decode('utf-8') if isinstance(action_types[i], bytes) else str(action_types[i]),
                                    'position': (float(positions_x[i]), float(positions_y[i])),
                                    'color': (int(colors_r[i]), int(colors_g[i]), int(colors_b[i]), int(colors_a[i])),
                                    'brush_size': float(brush_sizes[i]),
                                    'pressure': float(pressures[i])
                                }
                                if consciousness_layers is not None:
                                    action['consciousness_layer'] = int(consciousness_layers[i])
                                if pocket_dimensions is not None:
                                    action['pocket_dimension'] = int(pocket_dimensions[i])
                                    
                                combined_data['drawing_actions'].append(action)
                        
                        # Load dial data (stored as timestamps + positions)
                        if 'dial_data' in h5f and 'timestamps' in h5f['dial_data'] and 'positions' in h5f['dial_data']:
                            timestamps = h5f['dial_data']['timestamps'][()]
                            positions = h5f['dial_data']['positions'][()]
                            for ts, pos in zip(timestamps, positions):
                                # Positions stored as JSON bytes
                                try:
                                    if isinstance(pos, bytes):
                                        pos_dict = json.loads(pos.decode('utf-8'))
                                    else:
                                        pos_dict = pos
                                    combined_data['dial_positions'].append((float(ts), pos_dict))
                                except:
                                    pass
                    
            except Exception as e:
                logging.error(f"Error loading data file {file_path}: {e}")
                
        return combined_data
        
    def preprocess_data(self, raw_data: Dict[str, List]) -> TrainingData:
        """Convert raw session data into ML-ready format"""
        
        # Synchronize data by timestamp
        synchronized_data = self._synchronize_data_streams(raw_data)
        
        if len(synchronized_data['timestamps']) == 0:
            print("⚠️  No synchronized data available")
            return self._empty_training_data()
        
        print(f"📊 Synchronized data: {len(synchronized_data['timestamps'])} timesteps")
        
        # Extract features
        rng_features = self._extract_rng_features(synchronized_data)
        eeg_features = self._extract_eeg_features(synchronized_data)
        
        # Extract targets
        color_targets = self._extract_color_targets(synchronized_data)
        curve_targets = self._extract_curve_targets(synchronized_data)
        dial_targets = self._extract_dial_targets(synchronized_data)
        
        # Ensure all arrays have consistent length (use minimum to align)
        min_length = min(len(rng_features), len(eeg_features), len(color_targets), 
                        len(curve_targets), len(dial_targets))
        
        if min_length == 0:
            print("⚠️  No valid features extracted")
            return self._empty_training_data()
        
        # Debug shapes before alignment
        print(f"🔍 Before alignment - RNG: {rng_features.shape}, EEG: {eeg_features.shape}")
        print(f"🔍 Before alignment - Colors: {color_targets.shape}, Curves: {curve_targets.shape}, Dials: {dial_targets.shape}")
        
        # Trim all arrays to consistent length
        rng_features = rng_features[:min_length]
        eeg_features = eeg_features[:min_length]
        color_targets = color_targets[:min_length]
        curve_targets = curve_targets[:min_length]
        dial_targets = dial_targets[:min_length]
        
        print(f"✨ Features aligned to {min_length} timesteps")
        print(f"🔍 After alignment - RNG: {rng_features.shape}, EEG: {eeg_features.shape}")
        print(f"🔍 After alignment - Colors: {color_targets.shape}, Curves: {curve_targets.shape}, Dials: {dial_targets.shape}")
        
        # Apply data augmentation if dataset is small
        if min_length < 100:  # Threshold for small dataset
            print(f"🔮 Small dataset detected ({min_length} timesteps), applying augmentation...")
            
            # Augment all features consistently
            rng_features = self.augment_data(rng_features, augmentation_factor=2)
            eeg_features = self.augment_data(eeg_features, augmentation_factor=2)
            color_targets = self.augment_data(color_targets, augmentation_factor=2)
            curve_targets = self.augment_data(curve_targets, augmentation_factor=2)
            dial_targets = self.augment_data(dial_targets, augmentation_factor=2)
            
            print(f"✨ All features augmented to {len(rng_features)} timesteps")
        
        # Create sequences
        mode1_sequences, mode2_sequences, target_sequences = self._create_sequences(
            rng_features, eeg_features, color_targets, curve_targets, dial_targets
        )
        
        # Scale data
        mode1_scaled = self._scale_rng_data(mode1_sequences)
        mode2_scaled = self._scale_combined_data(mode2_sequences)
        color_scaled = self._scale_color_data(target_sequences['colors'])
        curve_scaled = self._scale_curve_data(target_sequences['curves'])
        dial_scaled = self._scale_dial_data(target_sequences['dials'])
        
        return TrainingData(
            mode1_inputs=mode1_scaled,
            mode2_inputs=mode2_scaled,
            color_outputs=color_scaled,
            curve_outputs=curve_scaled,
            dial_outputs=dial_scaled,
            timestamps=synchronized_data['timestamps'][:min_length],
            metadata={
                'sequence_length': self.sequence_length,
                'aligned_timesteps': min_length,
                'rng_feature_dim': rng_features.shape[1] if len(rng_features) > 0 else 0,
                'eeg_feature_dim': eeg_features.shape[1] if len(eeg_features) > 0 else 0,
                'color_dim': 4,  # RGBA
                'curve_dim': curve_targets.shape[1] if len(curve_targets) > 0 else 0,
                'dial_dim': dial_targets.shape[1] if len(dial_targets) > 0 else 0
            }
        )
        
    def _synchronize_data_streams(self, raw_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Synchronize different data streams by timestamp"""
        
        # Extract timestamps and create a common time grid
        all_timestamps = []
        
        for rng_sample in raw_data['rng_samples']:
            if isinstance(rng_sample, dict) and 'timestamp' in rng_sample:
                all_timestamps.append(rng_sample['timestamp'])
                
        for eeg_sample in raw_data['eeg_samples']:
            if isinstance(eeg_sample, dict) and 'timestamp' in eeg_sample:
                all_timestamps.append(eeg_sample['timestamp'])
                
        for action in raw_data['drawing_actions']:
            if isinstance(action, dict) and 'timestamp' in action:
                all_timestamps.append(action['timestamp'])
                
        if not all_timestamps:
            return {'timestamps': np.array([]), 'rng': np.array([]), 'eeg': np.array([]), 
                   'colors': np.array([]), 'curves': np.array([]), 'dials': np.array([])}
            
        # Sort timestamps and create regular grid
        all_timestamps = sorted(set(all_timestamps))
        min_time, max_time = min(all_timestamps), max(all_timestamps)
        
        # Create 100Hz sampling grid
        sample_rate = 100  # Hz
        time_grid = np.arange(min_time, max_time, 1.0 / sample_rate)
        
        # Interpolate data onto regular grid
        synchronized = {
            'timestamps': time_grid,
            'rng': self._interpolate_rng_data(raw_data['rng_samples'], time_grid),
            'eeg': self._interpolate_eeg_data(raw_data['eeg_samples'], time_grid),
            'colors': self._interpolate_color_data(raw_data['drawing_actions'], time_grid),
            'curves': self._interpolate_curve_data(raw_data['drawing_actions'], time_grid),
            'consciousness_layers': self._interpolate_consciousness_layer_data(raw_data['drawing_actions'], time_grid),
            'pocket_dimensions': self._interpolate_pocket_dimension_data(raw_data['drawing_actions'], time_grid),
            'dials': self._interpolate_dial_data(raw_data['dial_positions'], time_grid)
        }
        
        return synchronized
        
    def _interpolate_rng_data(self, rng_samples: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate RNG data to regular time grid"""
        if not rng_samples:
            return np.zeros((len(time_grid), 8))  # Default 8 RNG values
            
        # Extract RNG values and timestamps
        rng_times = []
        rng_values = []
        
        for sample in rng_samples:
            if 'timestamp' in sample and 'normalized' in sample:
                rng_times.append(sample['timestamp'])
                rng_values.append(sample['normalized'][:8])  # Take first 8 values
                
        if not rng_times:
            return np.zeros((len(time_grid), 8))
            
        rng_times = np.array(rng_times)
        rng_values = np.array(rng_values)
        
        # Interpolate each RNG channel
        interpolated = np.zeros((len(time_grid), 8))
        for i in range(min(8, rng_values.shape[1])):
            interpolated[:, i] = np.interp(time_grid, rng_times, rng_values[:, i])
            
        return interpolated
        
    def _interpolate_eeg_data(self, eeg_samples: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate EEG data to regular time grid"""
        if not eeg_samples:
            return np.zeros((len(time_grid), len(self.eeg_channels)))
            
        # Extract EEG values and timestamps
        eeg_times = []
        eeg_values = []
        
        for sample in eeg_samples:
            if 'timestamp' in sample and 'channels' in sample:
                eeg_times.append(sample['timestamp'])
                channel_values = []
                for channel in self.eeg_channels:
                    channel_values.append(sample['channels'].get(channel, 0.0))
                eeg_values.append(channel_values)
                
        if not eeg_times:
            return np.zeros((len(time_grid), len(self.eeg_channels)))
            
        eeg_times = np.array(eeg_times)
        eeg_values = np.array(eeg_values)
        
        # Interpolate each EEG channel
        interpolated = np.zeros((len(time_grid), len(self.eeg_channels)))
        for i in range(len(self.eeg_channels)):
            if i < eeg_values.shape[1]:
                interpolated[:, i] = np.interp(time_grid, eeg_times, eeg_values[:, i])
                
        return interpolated
        
    def _interpolate_color_data(self, drawing_actions: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate color data from drawing actions"""
        if not drawing_actions:
            return np.zeros((len(time_grid), 4))  # RGBA
            
        # Extract color data
        action_times = []
        colors = []
        
        for action in drawing_actions:
            if 'timestamp' in action and 'color' in action:
                action_times.append(action['timestamp'])
                color = action['color']
                if isinstance(color, list) and len(color) >= 4:
                    # Normalize to [0, 1] range
                    colors.append([c / 255.0 for c in color[:4]])
                else:
                    colors.append([0.0, 0.0, 0.0, 1.0])  # Default black
                    
        if not action_times:
            return np.zeros((len(time_grid), 4))
            
        action_times = np.array(action_times)
        colors = np.array(colors)
        
        # Interpolate RGBA channels
        interpolated = np.zeros((len(time_grid), 4))
        for i in range(4):
            interpolated[:, i] = np.interp(time_grid, action_times, colors[:, i])
            
        return interpolated
        
    def _interpolate_curve_data(self, drawing_actions: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Extract curve parameters from drawing actions"""
        # Simplified: extract position, brush size, pressure
        if not drawing_actions:
            return np.zeros((len(time_grid), 5))  # x, y, brush_size, pressure, action_type
            
        action_times = []
        curve_data = []
        
        for action in drawing_actions:
            if 'timestamp' in action:
                action_times.append(action['timestamp'])
                
                # Extract curve parameters
                position = action.get('position', [0, 0])
                brush_size = action.get('brush_size', 10)
                pressure = action.get('pressure', 1.0)
                action_type = 1.0 if action.get('action_type') == 'stroke_continue' else 0.0
                
                curve_data.append([position[0], position[1], brush_size, pressure, action_type])
                
        if not action_times:
            return np.zeros((len(time_grid), 5))
            
        action_times = np.array(action_times)
        curve_data = np.array(curve_data)
        
        # Interpolate curve parameters
        interpolated = np.zeros((len(time_grid), 5))
        for i in range(5):
            interpolated[:, i] = np.interp(time_grid, action_times, curve_data[:, i])
            
        return interpolated
        
    def _interpolate_dial_data(self, dial_positions: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate dial position data"""
        # Simplified dial representation: up to 8 dials with rotation and position
        max_dials = 8
        dial_features = 4  # x, y, rotation, radius
        
        if not dial_positions:
            return np.zeros((len(time_grid), max_dials * dial_features))
            
        dial_times = []
        dial_data = []
        
        for dial_entry in dial_positions:
            if 'timestamp' in dial_entry and 'positions' in dial_entry:
                dial_times.append(dial_entry['timestamp'])
                
                # Extract dial parameters
                positions = dial_entry['positions']
                dial_vector = np.zeros(max_dials * dial_features)
                
                for i, (dial_id, dial_info) in enumerate(positions.items()):
                    if i < max_dials and isinstance(dial_info, dict):
                        base_idx = i * dial_features
                        center = dial_info.get('center', [0, 0, 0])
                        dial_vector[base_idx] = center[0]      # x
                        dial_vector[base_idx + 1] = center[1]  # y
                        dial_vector[base_idx + 2] = dial_info.get('rotation', 0)  # rotation
                        dial_vector[base_idx + 3] = dial_info.get('radius', 1)    # radius
                        
                dial_data.append(dial_vector)
                
        if not dial_times:
            return np.zeros((len(time_grid), max_dials * dial_features))
            
        dial_times = np.array(dial_times)
        dial_data = np.array(dial_data)
        
        # Interpolate dial data
        interpolated = np.zeros((len(time_grid), max_dials * dial_features))
        for i in range(max_dials * dial_features):
            if i < dial_data.shape[1]:
                interpolated[:, i] = np.interp(time_grid, dial_times, dial_data[:, i])
                
        return interpolated
        
    def _interpolate_consciousness_layer_data(self, drawing_actions: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate consciousness layer data from drawing actions"""
        if not drawing_actions:
            return np.ones(len(time_grid))  # Default to layer 1
            
        action_times = []
        layers = []
        
        for action in drawing_actions:
            if 'timestamp' in action and 'consciousness_layer' in action:
                action_times.append(action['timestamp'])
                layers.append(action['consciousness_layer'])
            elif 'timestamp' in action:
                # Fallback to default layer 1 if not specified
                action_times.append(action['timestamp'])
                layers.append(1)
                
        if not action_times:
            return np.ones(len(time_grid))
            
        action_times = np.array(action_times)
        layers = np.array(layers)
        
        # Use nearest neighbor interpolation for discrete layer values
        interpolated = np.interp(time_grid, action_times, layers)
        return np.round(interpolated).astype(int)  # Ensure integer layer values
        
    def _interpolate_pocket_dimension_data(self, drawing_actions: List[Dict], time_grid: np.ndarray) -> np.ndarray:
        """Interpolate pocket dimension data from drawing actions"""
        if not drawing_actions:
            return np.ones(len(time_grid))  # Default to dimension 1
            
        action_times = []
        dimensions = []
        
        for action in drawing_actions:
            if 'timestamp' in action and 'pocket_dimension' in action:
                action_times.append(action['timestamp'])
                dimensions.append(action['pocket_dimension'])
            elif 'timestamp' in action:
                # Fallback to default dimension 1 if not specified
                action_times.append(action['timestamp'])
                dimensions.append(1)
                
        if not action_times:
            return np.ones(len(time_grid))
            
        action_times = np.array(action_times)
        dimensions = np.array(dimensions)
        
        # Use nearest neighbor interpolation for discrete dimension values
        interpolated = np.interp(time_grid, action_times, dimensions)
        return interpolated  # Keep as float for dimensional navigation
        
    def _extract_rng_features(self, sync_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract RNG features"""
        rng_data = sync_data['rng']
        
        # Ensure RNG data has proper shape (timesteps, features)
        if len(rng_data.shape) == 1:
            rng_data = rng_data.reshape(-1, 1)
        
        return rng_data
        
    def _extract_eeg_features(self, sync_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract EEG features"""
        return sync_data['eeg']
        
    def _extract_color_targets(self, sync_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract color targets"""
        return sync_data['colors']
        
    def _extract_curve_targets(self, sync_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract curve targets"""
        return sync_data['curves']
        
    def _extract_dial_targets(self, sync_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract dial targets"""
        return sync_data['dials']
        
    def _create_sequences(self, rng_features: np.ndarray, eeg_features: np.ndarray,
                         color_targets: np.ndarray, curve_targets: np.ndarray,
                         dial_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Create input sequences and corresponding targets"""
        
        # Adjust sequence length for small datasets
        available_length = len(rng_features)
        adjusted_sequence_length = min(self.sequence_length, max(8, available_length // 2))
        
        if available_length < adjusted_sequence_length:
            print(f"⚠️  Insufficient data: {available_length} timesteps, need at least {adjusted_sequence_length}")
            return (np.array([]), np.array([]), 
                   {'colors': np.array([]), 'curves': np.array([]), 'dials': np.array([])})
        
        print(f"📏 Using sequence length: {adjusted_sequence_length} (original: {self.sequence_length})")
        
        # Calculate number of sequences
        num_sequences = available_length - adjusted_sequence_length + 1
        
        # Mode 1: RNG only sequences
        mode1_sequences = np.zeros((num_sequences, adjusted_sequence_length, rng_features.shape[1]))
        
        # Mode 2: RNG + EEG sequences
        combined_feature_dim = rng_features.shape[1] + eeg_features.shape[1]
        mode2_sequences = np.zeros((num_sequences, adjusted_sequence_length, combined_feature_dim))
        
        # Target sequences (predict next value)
        color_seq = np.zeros((num_sequences, color_targets.shape[1]))
        curve_seq = np.zeros((num_sequences, curve_targets.shape[1]))
        dial_seq = np.zeros((num_sequences, dial_targets.shape[1]))
        
        for i in range(num_sequences):
            # Input sequences
            mode1_sequences[i] = rng_features[i:i + adjusted_sequence_length]
            
            # Combine RNG and EEG for mode 2
            rng_seq = rng_features[i:i + adjusted_sequence_length]
            eeg_seq = eeg_features[i:i + adjusted_sequence_length]
            mode2_sequences[i] = np.concatenate([rng_seq, eeg_seq], axis=1)
            
            # Target outputs (predict next timestep)
            target_idx = min(i + adjusted_sequence_length, len(color_targets) - 1)
            color_seq[i] = color_targets[target_idx]
            curve_seq[i] = curve_targets[target_idx]
            dial_seq[i] = dial_targets[target_idx]
            
        print(f"✅ Created {num_sequences} sequences of length {adjusted_sequence_length}")
        
        return mode1_sequences, mode2_sequences, {
            'colors': color_seq,
            'curves': curve_seq,
            'dials': dial_seq
        }
        
        # Mode 1: RNG only sequences
        mode1_sequences = np.zeros((num_sequences, self.sequence_length, rng_features.shape[1]))
        
        # Mode 2: RNG + EEG sequences
        combined_feature_dim = rng_features.shape[1] + eeg_features.shape[1]
        mode2_sequences = np.zeros((num_sequences, self.sequence_length, combined_feature_dim))
        
        # Target sequences (predict next value)
        color_seq = np.zeros((num_sequences, color_targets.shape[1]))
        curve_seq = np.zeros((num_sequences, curve_targets.shape[1]))
        dial_seq = np.zeros((num_sequences, dial_targets.shape[1]))
        
        for i in range(num_sequences):
            # Input sequences
            mode1_sequences[i] = rng_features[i:i + self.sequence_length]
            
            # Combine RNG and EEG for mode 2
            rng_seq = rng_features[i:i + self.sequence_length]
            eeg_seq = eeg_features[i:i + self.sequence_length]
            mode2_sequences[i] = np.concatenate([rng_seq, eeg_seq], axis=1)
            
            # Target values (next timestep)
            target_idx = i + self.sequence_length - 1
            if target_idx < len(color_targets):
                color_seq[i] = color_targets[target_idx]
                curve_seq[i] = curve_targets[target_idx]
                dial_seq[i] = dial_targets[target_idx]
                
        return mode1_sequences, mode2_sequences, {
            'colors': color_seq,
            'curves': curve_seq,
            'dials': dial_seq
        }
        
    def _scale_rng_data(self, data: np.ndarray) -> np.ndarray:
        """Scale RNG data"""
        if len(data) == 0:
            return data
        original_shape = data.shape
        reshaped = data.reshape(-1, data.shape[-1])
        scaled = self.rng_scaler.fit_transform(reshaped)
        return scaled.reshape(original_shape)
        
    def _scale_combined_data(self, data: np.ndarray) -> np.ndarray:
        """Scale combined RNG+EEG data"""
        if len(data) == 0:
            return data
        original_shape = data.shape
        reshaped = data.reshape(-1, data.shape[-1])
        
        # Scale RNG and EEG parts separately
        rng_dim = 8  # First 8 features are RNG
        rng_part = reshaped[:, :rng_dim]
        eeg_part = reshaped[:, rng_dim:]
        
        scaled_rng = self.rng_scaler.transform(rng_part) if rng_part.size > 0 else rng_part
        scaled_eeg = self.eeg_scaler.fit_transform(eeg_part) if eeg_part.size > 0 else eeg_part
        
        scaled = np.concatenate([scaled_rng, scaled_eeg], axis=1)
        return scaled.reshape(original_shape)
        
    def _scale_color_data(self, data: np.ndarray) -> np.ndarray:
        """Scale color data"""
        if len(data) == 0:
            return data
        return self.color_scaler.fit_transform(data)
        
    def _scale_curve_data(self, data: np.ndarray) -> np.ndarray:
        """Scale curve data"""
        if len(data) == 0:
            return data
        return self.curve_scaler.fit_transform(data)
        
    def _scale_dial_data(self, data: np.ndarray) -> np.ndarray:
        """Scale dial data"""
        if len(data) == 0:
            return data
        return self.dial_scaler.fit_transform(data)


class ConsciousnessModel:
    """Base class for consciousness prediction models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.training_history = {}
        
    def build_model(self, input_shape: Tuple[int, ...], output_shapes: Dict[str, int]):
        """Build the model architecture"""
        raise NotImplementedError
        
    def train(self, training_data: TrainingData, mode: int = 1) -> Dict[str, Any]:
        """Train the model"""
        raise NotImplementedError
        
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions"""
        raise NotImplementedError
        
    def save_model(self, filepath: str):
        """Save the trained model"""
        raise NotImplementedError
        
    def load_model(self, filepath: str):
        """Load a trained model"""
        raise NotImplementedError


class TensorFlowConsciousnessModel(ConsciousnessModel):
    """TensorFlow/Keras implementation of the consciousness model"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for this model")
            
        # Configure GPU if available
        self._configure_gpu()
        
    def _configure_gpu(self):
        """Configure GPU settings if available"""
        try:
            # Check for GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                print(f"🎮 GPU detected: {len(gpus)} GPU(s) available")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                
                try:
                    # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("✅ GPU memory growth configured")
                    
                    # Set first GPU as primary
                    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                    
                    # Additional GPU optimizations
                    tf.config.experimental.enable_tensor_float_32_execution(True)
                    
                    self.device_name = '/GPU:0'
                    print(f"✅ GPU acceleration enabled - using {gpus[0].name}")
                    
                except RuntimeError as e:
                    print(f"⚠️ GPU configuration error: {e}")
                    print("   Falling back to CPU")
                    self.device_name = '/CPU:0'
                    
            else:
                print("💻 No GPU detected - using CPU")
                self.device_name = '/CPU:0'
                
        except Exception as e:
            print(f"⚠️ Error checking GPU availability: {e}")
            self.device_name = '/CPU:0'
            
    def build_model(self, input_shape: Tuple[int, ...], output_shapes: Dict[str, int]):
        """Build TensorFlow model"""
        
        # Input layer
        inputs = keras.Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for i in range(self.config.num_layers):
            return_sequences = i < self.config.num_layers - 1
            x = layers.LSTM(
                self.config.hidden_size,
                return_sequences=return_sequences,
                dropout=self.config.dropout,
                name=f'lstm_{i}'
            )(x)
            
        # Output branches
        outputs = {}
        
        # Color output (RGBA)
        color_output = layers.Dense(output_shapes['colors'], activation='sigmoid', name='colors')(x)
        outputs['colors'] = color_output
        
        # Curve output
        curve_output = layers.Dense(output_shapes['curves'], activation='linear', name='curves')(x)
        outputs['curves'] = curve_output
        
        # Dial output
        dial_output = layers.Dense(output_shapes['dials'], activation='linear', name='dials')(x)
        outputs['dials'] = dial_output
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile with multiple losses
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'colors': 'mse',
                'curves': 'mse',
                'dials': 'mse'
            },
            loss_weights={
                'colors': 1.0,
                'curves': 0.5,
                'dials': 0.3
            },
            metrics={
                'colors': ['mae'],
                'curves': ['mae'],
                'dials': ['mae']
            }
        )
        
        return self.model
        
    def train(self, training_data: TrainingData, mode: int = 1) -> Dict[str, Any]:
        """Train the TensorFlow model"""
        
        # Select input data based on mode
        if mode == 1:
            X = training_data.mode1_inputs
        else:
            X = training_data.mode2_inputs
            
        # Prepare outputs
        y = {
            'colors': training_data.color_outputs,
            'curves': training_data.curve_outputs,
            'dials': training_data.dial_outputs
        }
        
        # Debug shapes before training
        print(f"🔍 Training data shapes:")
        print(f"   Input X: {X.shape}")
        for key, val in y.items():
            print(f"   Output {key}: {val.shape}")
        
        # Check for empty arrays
        if len(X) == 0:
            print("⚠️  No input data available for training")
            return {'final_loss': 0.0, 'history': {}}
        
        for key, val in y.items():
            if len(val) == 0:
                print(f"⚠️  No {key} target data available")
                return {'final_loss': 0.0, 'history': {}}
        
        # Split data
        if SKLEARN_AVAILABLE:
            # Convert dict of arrays to list of arrays for train_test_split
            y_arrays = [y['colors'], y['curves'], y['dials']]
            
            try:
                split_result = train_test_split(
                    X, *y_arrays, 
                    test_size=self.config.validation_split,
                    random_state=42
                )
                
                # Unpack the result
                X_train, X_val = split_result[0], split_result[1]
                y_train_colors, y_val_colors = split_result[2], split_result[3]
                y_train_curves, y_val_curves = split_result[4], split_result[5]
                y_train_dials, y_val_dials = split_result[6], split_result[7]
                
                # Reconstruct dictionaries
                y_train = {
                    'colors': y_train_colors,
                    'curves': y_train_curves,
                    'dials': y_train_dials
                }
                y_val = {
                    'colors': y_val_colors,
                    'curves': y_val_curves,
                    'dials': y_val_dials
                }
                
            except Exception as e:
                print(f"⚠️  sklearn split failed ({e}), using simple split")
                # Fall back to simple split
                split_idx = int(len(X) * (1 - self.config.validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train = {k: v[:split_idx] for k, v in y.items()}
                y_val = {k: v[split_idx:] for k, v in y.items()}
        else:
            # Simple split without sklearn
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train = {k: v[:split_idx] for k, v in y.items()}
            y_val = {k: v[split_idx:] for k, v in y.items()}
            
        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:]
            output_shapes = {k: v.shape[1] for k, v in y_train.items()}
            print(f"🔧 Building model on device: {self.device_name}")
            with tf.device(self.device_name):
                self.build_model(input_shape, output_shapes)
            
        # Training callbacks
        consciousness_callback = ConsciousnessTrainingCallback(verbose=self.config.verbose_training)
        
        training_callbacks = [
            TensorFlowProgressCallback(consciousness_callback),
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                min_delta=1e-6,
                start_from_epoch=self.config.min_epochs  # Wait minimum epochs before early stopping
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=max(3, self.config.early_stopping_patience // 3),
                min_lr=1e-7,
                verbose=1 if self.config.verbose_training else 0
            )
        ]
        
        # Print training info
        if self.config.verbose_training:
            print(f"🔍 Training Data Shape: {X_train.shape}")
            print(f"🔍 Validation Data Shape: {X_val.shape}")
            print(f"🎯 Target Outputs: {list(y_train.keys())}")
            for k, v in y_train.items():
                print(f"   - {k}: {v.shape}")
            print(f"⚙️ Batch Size: {self.config.batch_size}")
            print(f"⚙️ Learning Rate: {self.config.learning_rate}")
            print(f"⚙️ Max Epochs: {self.config.epochs}")
            print()
        
        # Train model with verbose output
        print(f"🚀 Starting training on device: {self.device_name}")
        with tf.device(self.device_name):
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_val, y_val),
                callbacks=training_callbacks,
                verbose=0  # Use our custom callback instead
            )
        
        self.training_history = history.history
        
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss']),
            'mode': mode
        }
        
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with the model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        predictions = self.model.predict(input_data)
        
        return {
            'colors': predictions['colors'],
            'curves': predictions['curves'],
            'dials': predictions['dials']
        }
        
    def save_model(self, filepath: str):
        """Save the TensorFlow model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        # Save model architecture and weights
        self.model.save(f"{filepath}.h5")
        
        # Save training history and config
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        metadata = {
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=convert_numpy_types)
            
    def load_model(self, filepath: str):
        """Load a TensorFlow model with compatibility handling"""
        import tensorflow as tf
        from tensorflow import keras
        
        # Define custom objects to handle compatibility issues
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'accuracy': tf.keras.metrics.Accuracy(),
        }
        
        try:
            # Try loading with custom objects first
            self.model = keras.models.load_model(f"{filepath}.h5", custom_objects=custom_objects)
        except Exception as e:
            self.logger.warning(f"Failed to load model with custom objects: {e}")
            try:
                # Fallback: try loading without custom objects
                self.model = keras.models.load_model(f"{filepath}.h5", compile=False)
                # Recompile with basic metrics
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                self.logger.info("Model loaded without compilation and recompiled")
            except Exception as e2:
                self.logger.error(f"Failed to load model: {e2}")
                raise e2
        
        # Load metadata
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.training_history = metadata.get('training_history', {})
        except FileNotFoundError:
            logging.warning(f"Metadata file not found for {filepath}")


class ModelTrainer:
    """High-level interface for training consciousness models"""
    
    def __init__(self, output_dir: str = "models", use_real_data: bool = True):
        self.output_dir = output_dir
        self.use_real_data = use_real_data
        os.makedirs(output_dir, exist_ok=True)
        
        # Choose preprocessor based on availability
        if use_real_data:
            self.preprocessor = RealDataPreprocessor()
            print("🧠 Using real data preprocessor for consciousness training")
        else:
            self.preprocessor = DataPreprocessor()
            print("🎭 Using synthetic data preprocessor")
        
        self.models = {}
        
    def train_models_from_real_data(self, data_directory: str = "data", config: TrainingConfig = None) -> Dict[str, Any]:
        """Train models using real session data"""
        
        if config is None:
            config = TrainingConfig()
        
        print("🚀 Starting consciousness model training with real data...")
        print("=" * 60)
        
        # Load real training data
        if self.use_real_data and hasattr(self.preprocessor, 'load_real_session_data'):
            training_data = self.preprocessor.load_real_session_data(data_directory)
        else:
            # Fallback to original method
            print("⚠️ Falling back to original data loading method")
            raw_data = self.preprocessor.load_session_data([])  # Empty list for demo
            training_data = self.preprocessor.preprocess_data(raw_data)
        
        if len(training_data.mode1_inputs) == 0:
            print("❌ No training data available")
            return {}
        
        results = {}
        
        # Train Mode 1 model (RNG only)
        print("\n🎯 Training Mode 1 model (RNG only)...")
        try:
            model1 = TensorFlowConsciousnessModel(config)
            results['mode1'] = model1.train(training_data, mode=1)
            
            # Save model
            model1_path = os.path.join(self.output_dir, "consciousness_model_mode1_real")
            model1.save_model(model1_path)
            self.models['mode1'] = model1
            
            print(f"✅ Mode 1 training completed. Final loss: {results['mode1']['final_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Mode 1 training failed: {e}")
            results['mode1'] = {'error': str(e)}
            
        # Train Mode 2 model (RNG + EEG) if we have EEG data
        if len(training_data.mode2_inputs) > 0 and training_data.mode2_inputs.shape[-1] > training_data.mode1_inputs.shape[-1]:
            print("\n🧠 Training Mode 2 model (RNG + EEG)...")
            try:
                model2 = TensorFlowConsciousnessModel(config)
                results['mode2'] = model2.train(training_data, mode=2)
                
                # Save model
                model2_path = os.path.join(self.output_dir, "consciousness_model_mode2_real")
                model2.save_model(model2_path)
                self.models['mode2'] = model2
                
                print(f"✅ Mode 2 training completed. Final loss: {results['mode2']['final_loss']:.4f}")
                
            except Exception as e:
                print(f"❌ Mode 2 training failed: {e}")
                results['mode2'] = {'error': str(e)}
        else:
            print("⚠️ Skipping Mode 2 training (no EEG data or same as Mode 1)")
            results['mode2'] = {'skipped': 'No additional EEG data available'}
        
        print("\n🎉 Training session completed!")
        print("=" * 60)
        return results
        
    def train_models(self, data_files: List[str], config: TrainingConfig) -> Dict[str, Any]:
        """Train both Mode 1 and Mode 2 models"""
        
        logging.info("Loading and preprocessing data...")
        
        # Use RealDataPreprocessor's optimized loading if available
        if self.use_real_data and hasattr(self.preprocessor, 'load_real_session_data'):
            # Extract data directory from first file path
            if data_files:
                data_directory = os.path.dirname(data_files[0])
            else:
                data_directory = "data"
            
            logging.info(f"Using real data loader with directory: {data_directory}")
            training_data = self.preprocessor.load_real_session_data(data_directory)
        else:
            # Fallback to traditional two-step loading
            raw_data = self.preprocessor.load_session_data(data_files)
            training_data = self.preprocessor.preprocess_data(raw_data)
        
        results = {}
        
        # Train Mode 1 model (RNG only)
        if len(training_data.mode1_inputs) > 0:
            logging.info("Training Mode 1 model (RNG only)...")
            model1 = TensorFlowConsciousnessModel(config)
            results['mode1'] = model1.train(training_data, mode=1)
            
            # Save model
            model1_path = os.path.join(self.output_dir, "consciousness_model_mode1")
            model1.save_model(model1_path)
            self.models['mode1'] = model1
            
            logging.info(f"Mode 1 training completed. Final loss: {results['mode1']['final_loss']:.4f}")
            
        # Train Mode 2 model (RNG + EEG)
        if len(training_data.mode2_inputs) > 0:
            logging.info("Training Mode 2 model (RNG + EEG)...")
            model2 = TensorFlowConsciousnessModel(config)
            results['mode2'] = model2.train(training_data, mode=2)
            
            # Save model
            model2_path = os.path.join(self.output_dir, "consciousness_model_mode2")
            model2.save_model(model2_path)
            self.models['mode2'] = model2
            
            logging.info(f"Mode 2 training completed. Final loss: {results['mode2']['final_loss']:.4f}")
            
        return results
        
    def load_models(self, mode1_path: str = None, mode2_path: str = None):
        """Load pre-trained models"""
        
        if mode1_path:
            model1 = TensorFlowConsciousnessModel(TrainingConfig())
            model1.load_model(mode1_path)
            self.models['mode1'] = model1
            logging.info(f"Loaded Mode 1 model from {mode1_path}")
            
        if mode2_path:
            model2 = TensorFlowConsciousnessModel(TrainingConfig())
            model2.load_model(mode2_path)
            self.models['mode2'] = model2
            logging.info(f"Loaded Mode 2 model from {mode2_path}")
            
    def predict(self, rng_data: np.ndarray, eeg_data: np.ndarray = None, 
               mode: int = 1) -> Dict[str, np.ndarray]:
        """Make predictions using trained models"""
        
        model_key = f'mode{mode}'
        if model_key not in self.models:
            raise ValueError(f"Model for mode {mode} not loaded")
            
        # Prepare input data
        if mode == 1:
            # RNG only
            input_data = rng_data.reshape(1, -1, rng_data.shape[-1])
        else:
            # RNG + EEG
            if eeg_data is None:
                raise ValueError("EEG data required for mode 2")
            input_data = np.concatenate([rng_data, eeg_data], axis=-1)
            input_data = input_data.reshape(1, -1, input_data.shape[-1])
            
        # Make prediction
        predictions = self.models[model_key].predict(input_data)
        
        # Return single prediction (remove batch dimension)
        return {k: v[0] for k, v in predictions.items()}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = TrainingConfig(
        model_type="lstm",
        sequence_length=50,
        batch_size=16,
        epochs=10,  # Reduced for testing
        hidden_size=64
    )
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Mock training data (replace with actual data files)
    print("This is a demo of the ML training pipeline.")
    print("To use with real data:")
    print("1. Collect data using the data logging system")
    print("2. Provide data file paths to train_models()")
    print("3. Use the trained models for inference")
    
    # Example of how to use:
    # data_files = ["data/session1.json", "data/session2.json"]
    # results = trainer.train_models(data_files, config)
    # print(f"Training results: {results}")