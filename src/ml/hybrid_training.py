#!/usr/bin/env python3
"""
Hybrid TensorFlow-PyTorch Consciousness Training Pipeline
Uses PyTorch for GPU acceleration while maintaining TensorFlow compatibility
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
import json
from datetime import datetime
import torch

# Add sklearn import for compatibility
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except ImportError:
    print("Warning: sklearn not installed. Some features may not work.")

# Import existing TensorFlow pipeline
from ml.training_pipeline import (
    TrainingConfig, 
    ConsciousnessTrainingCallback,
    DataPreprocessor,
    TrainingData
)

# Import our new PyTorch model
from ml.pytorch_consciousness_model import (
    create_pytorch_consciousness_model,
    PyTorchConsciousnessTrainer
)

class HybridConsciousnessTrainer:
    """
    Hybrid trainer that uses PyTorch GPU acceleration with TensorFlow data pipeline
    """
    
    def __init__(self, config: TrainingConfig, use_pytorch_gpu: bool = True):
        self.config = config
        self.use_pytorch_gpu = use_pytorch_gpu and torch.cuda.is_available()
        
        # Initialize data preprocessor (from TensorFlow pipeline)
        self.preprocessor = DataPreprocessor()
        
        # Setup training framework
        if self.use_pytorch_gpu:
            print("🚀 Using PyTorch GPU acceleration")
            self.framework = "pytorch"
            self.pytorch_trainer = None
        else:
            print("💻 Using TensorFlow CPU fallback")
            self.framework = "tensorflow"
            # Import TensorFlow model only if needed
            from ml.training_pipeline import TensorFlowConsciousnessModel
            self.tf_model = TensorFlowConsciousnessModel(config)
    
    def _prepare_data_for_pytorch(self, training_data: TrainingData) -> Tuple[
        Tuple[np.ndarray, Dict[str, np.ndarray]], 
        Tuple[np.ndarray, Dict[str, np.ndarray]]
    ]:
        """Convert TensorFlow training data format to PyTorch format"""
        
        # Extract features and targets
        X_train = training_data.X_train
        X_val = training_data.X_val
        
        # Combine all y outputs into dictionaries
        y_train = {}
        y_val = {}
        
        for key in training_data.y_train.keys():
            y_train[key] = training_data.y_train[key]
            y_val[key] = training_data.y_val[key]
        
        return (X_train, y_train), (X_val, y_val)
    
    def _convert_pytorch_to_tf_format(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PyTorch training history to TensorFlow format"""
        
        tf_history = {
            'loss': history['train_loss'],
            'val_loss': history['val_loss']
        }
        
        # Add individual output losses
        for output_name in history['train_losses'].keys():
            tf_history[f'{output_name}_loss'] = history['train_losses'][output_name]
            tf_history[f'val_{output_name}_loss'] = history['val_losses'][output_name]
        
        return tf_history
    
    def preprocess_data(self, data_files: list) -> TrainingData:
        """Use existing TensorFlow data preprocessing pipeline"""
        print("📊 Preprocessing consciousness data...")
        
        # Load and merge all data files
        all_data = {'rng_data': [], 'eeg_data': [], 'drawing_data': []}
        
        for file_path in data_files:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                    for key in all_data.keys():
                        if key in session_data:
                            all_data[key].extend(session_data[key])
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                continue
        
        # Use the existing preprocessor
        training_data = self.preprocessor.preprocess_data(all_data)
        
        print(f"✅ Data preprocessing completed")
        print(f"   Training samples: {len(training_data.X_train)}")
        print(f"   Validation samples: {len(training_data.X_val)}")
        print(f"   Feature dimensions: {training_data.X_train.shape[1:]}")
        print(f"   Output targets: {list(training_data.y_train.keys())}")
        
        return training_data
    
    def train(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train consciousness model using best available framework"""
        
        print("\n" + "=" * 80)
        print("🧠 HYBRID CONSCIOUSNESS MODEL TRAINING")
        print("=" * 80)
        print(f"🔧 Framework: {self.framework.upper()}")
        print(f"🎮 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"📊 Training Mode: {'GPU Accelerated' if self.use_pytorch_gpu else 'CPU Optimized'}")
        print()
        
        if self.framework == "pytorch":
            return self._train_with_pytorch(training_data)
        else:
            return self._train_with_tensorflow(training_data)
    
    def _train_with_pytorch(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train using PyTorch GPU acceleration"""
        
        # Convert data format
        train_data, val_data = self._prepare_data_for_pytorch(training_data)
        
        # Determine model architecture
        input_shape = train_data[0].shape[1:]
        output_shapes = {k: v.shape[1] for k, v in train_data[1].items()}
        
        print(f"🏗️  Building PyTorch model...")
        print(f"   Input shape: {input_shape}")
        print(f"   Output shapes: {output_shapes}")
        
        # Create PyTorch model
        self.pytorch_trainer = create_pytorch_consciousness_model(input_shape, output_shapes)
        
        # Train with PyTorch
        print("\n🚀 Starting PyTorch GPU training...")
        pytorch_history = self.pytorch_trainer.train(
            train_data, val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )
        
        # Convert to TensorFlow-compatible format
        tf_compatible_history = self._convert_pytorch_to_tf_format(pytorch_history)
        
        return {
            'framework': 'pytorch',
            'device': 'GPU' if torch.cuda.is_available() else 'CPU',
            'history': tf_compatible_history,
            'pytorch_history': pytorch_history,
            'model_path': 'best_consciousness_model_pytorch.pth'
        }
    
    def _train_with_tensorflow(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train using TensorFlow (fallback)"""
        
        print("🚀 Starting TensorFlow training...")
        
        # Use existing TensorFlow training
        tf_history = self.tf_model.train(
            training_data.X_train, training_data.y_train,
            training_data.X_val, training_data.y_val
        )
        
        return {
            'framework': 'tensorflow',
            'device': self.tf_model.device_name,
            'history': tf_history,
            'model': self.tf_model
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using the trained model"""
        
        if self.framework == "pytorch" and self.pytorch_trainer:
            return self.pytorch_trainer.predict(X)
        elif self.framework == "tensorflow":
            return self.tf_model.predict(X)
        else:
            raise RuntimeError("No trained model available for predictions")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if self.framework == "pytorch" and self.pytorch_trainer:
            self.pytorch_trainer.save_model(filepath)
        elif self.framework == "tensorflow":
            self.tf_model.save_model(filepath)
    
    def get_training_summary(self) -> str:
        """Get a summary of the training configuration"""
        
        summary = f"""
🧠 Hybrid Consciousness Training Configuration

Framework: {self.framework.upper()}
GPU Acceleration: {'✅ Enabled' if self.use_pytorch_gpu else '❌ Disabled'}
Device: {'GPU' if self.use_pytorch_gpu and torch.cuda.is_available() else 'CPU'}

Training Parameters:
- Epochs: {self.config.epochs}
- Batch Size: {self.config.batch_size}
- Learning Rate: {self.config.learning_rate}
- Validation Split: {self.config.validation_split}

Hardware Information:
"""
        
        if torch.cuda.is_available():
            summary += f"- GPU: {torch.cuda.get_device_name(0)}\n"
            summary += f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB\n"
        else:
            summary += "- GPU: Not available\n"
        
        summary += f"- PyTorch Version: {torch.__version__}\n"
        
        try:
            import tensorflow as tf
            summary += f"- TensorFlow Version: {tf.__version__}\n"
        except ImportError:
            summary += "- TensorFlow: Not available\n"
        
        return summary

def train_consciousness_hybrid(data_files: list, config: TrainingConfig = None, 
                             force_pytorch: bool = False, force_tensorflow: bool = False) -> Dict[str, Any]:
    """
    High-level function to train consciousness model with hybrid PyTorch/TensorFlow support
    
    Args:
        data_files: List of consciousness data files
        config: Training configuration
        force_pytorch: Force use of PyTorch even if GPU not available
        force_tensorflow: Force use of TensorFlow even if GPU available
    
    Returns:
        Training results dictionary
    """
    
    if config is None:
        config = TrainingConfig()
    
    # Determine framework preference
    use_pytorch_gpu = True
    if force_tensorflow:
        use_pytorch_gpu = False
    elif force_pytorch:
        use_pytorch_gpu = True
    
    # Create hybrid trainer
    trainer = HybridConsciousnessTrainer(config, use_pytorch_gpu)
    
    print(trainer.get_training_summary())
    
    # Preprocess data
    training_data = trainer.preprocess_data(data_files)
    
    # Train model
    results = trainer.train(training_data)
    
    # Add trainer reference to results
    results['trainer'] = trainer
    
    return results

if __name__ == "__main__":
    # Test hybrid training system
    print("=== Testing Hybrid PyTorch-TensorFlow Consciousness Training ===\n")
    
    # Check for existing data files
    data_dir = "../../data"
    if os.path.exists(data_dir):
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        
        if data_files:
            print(f"Found {len(data_files)} data files for training")
            
            # Test with PyTorch GPU
            config = TrainingConfig(epochs=20, batch_size=32)
            results = train_consciousness_hybrid(data_files[:3], config)  # Use first 3 files for testing
            
            print(f"\n✅ Training completed using {results['framework']}")
            print(f"🎮 Device: {results['device']}")
            print(f"📊 Final loss: {results['history']['loss'][-1]:.6f}")
        else:
            print("⚠️  No data files found in data directory")
            print("   Run data generation first: python run.py --test-rng --no-eeg")
    else:
        print("⚠️  Data directory not found")
        print("   Please run from consciousness-app directory")