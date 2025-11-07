"""
Advanced Inference Engine for Consciousness Models
Supports loading and comparing multiple model variants
"""

import logging
import queue
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .model_manager import ModelManager, ModelMetadata

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

@dataclass
class PredictionResult:
    """Result from model prediction"""
    timestamp: float
    colors: List[float]  # RGBA values
    curves: List[float]  # Curve parameters
    dials: List[float]   # Dial positions/rotations
    confidence: float
    model_name: str = ""
    input_features: List[str] = None

@dataclass
class MultiModelInferenceConfig:
    """Configuration for multi-model inference"""
    sequence_length: int = 50
    enable_gpu: bool = True
    auto_select_best: bool = True
    compare_models: bool = False
    max_models: int = 5  # Maximum number of models to load simultaneously
    
class ModelInstance:
    """Wrapper for a loaded model instance"""
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.model = None
        self.is_loaded = False
        self.last_used = None
        
    def load(self):
        """Load the model into memory"""
        try:
            if self.metadata.variant_config.framework == "pytorch" and PYTORCH_AVAILABLE:
                self._load_pytorch_model()
            elif self.metadata.variant_config.framework == "tensorflow" and TENSORFLOW_AVAILABLE:
                self._load_tensorflow_model()
            else:
                raise ImportError(f"Framework {self.metadata.variant_config.framework} not available")
            
            self.is_loaded = True
            logging.info(f"Loaded model: {self.metadata.variant_config.name}")
            
        except Exception as e:
            logging.error(f"Error loading model {self.metadata.variant_config.name}: {e}")
            raise
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        import torch
        import json
        from pathlib import Path
        
        model_dir = Path(self.metadata.model_path)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.pth"
        
        # Check if saved files exist
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        # Load config
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Create model architecture
        model = self._create_pytorch_model_from_config(saved_config)
        
        # Load weights
        device = 'cuda' if self.metadata.variant_config.use_gpu and torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        self.model = model
        logging.info(f"Loaded PyTorch model from {weights_path}")
    
    def _create_pytorch_model_from_config(self, config):
        """Recreate PyTorch model from saved configuration"""
        import torch
        import torch.nn as nn
        
        # Determine input size based on features
        input_size = 0
        if "rng" in config["input_features"]:
            input_size += 8  # RNG features
        if "eeg" in config["input_features"]:
            input_size += 32  # EEG features
        
        # Recreate the same model architectures as in training
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  dropout=dropout_rate, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()  # Constrain output to [0, 1]
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)  # Apply sigmoid activation
                return out
        
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                dropout=dropout_rate, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()  # Constrain output to [0, 1]
                
            def forward(self, x):
                out, _ = self.gru(x)
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)  # Apply sigmoid activation
                return out
        
        class TransformerModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_size, nhead=min(8, input_size), 
                    dim_feedforward=hidden_size, dropout=dropout_rate,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(input_size, 1)
                self.sigmoid = nn.Sigmoid()  # Constrain output to [0, 1]
                
            def forward(self, x):
                out = self.transformer(x)
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)  # Apply sigmoid activation
                return out
        
        class CNNLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
                super().__init__()
                self.conv1d = nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1)
                self.lstm = nn.LSTM(hidden_size//2, hidden_size, num_layers,
                                  dropout=dropout_rate, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()  # Constrain output to [0, 1]
                
            def forward(self, x):
                x = x.transpose(1, 2)
                x = torch.relu(self.conv1d(x))
                x = x.transpose(1, 2)
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)  # Apply sigmoid activation
                return out
        
        # Create model based on architecture
        if config["architecture"] == "lstm":
            model = LSTMModel(input_size, config["hidden_size"], 
                            config["num_layers"], config["dropout_rate"])
        elif config["architecture"] == "gru":
            model = GRUModel(input_size, config["hidden_size"],
                           config["num_layers"], config["dropout_rate"])
        elif config["architecture"] == "transformer":
            model = TransformerModel(input_size, config["hidden_size"],
                                   config["num_layers"], config["dropout_rate"])
        elif config["architecture"] == "cnn_lstm":
            model = CNNLSTMModel(input_size, config["hidden_size"],
                               config["num_layers"], config["dropout_rate"])
        else:
            raise ValueError(f"Unsupported architecture: {config['architecture']}")
        
        return model
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model"""
        from .training_pipeline import TensorFlowConsciousnessModel, TrainingConfig
        
        config = self.metadata.variant_config
        training_config = TrainingConfig(
            model_type=config.architecture,
            sequence_length=config.sequence_length,
            batch_size=config.batch_size,
            hidden_size=config.hidden_size
        )
        
        model = TensorFlowConsciousnessModel(training_config)
        model.load_model(self.metadata.model_path)
        
        self.model = model
    
    def predict(self, input_data: Any) -> PredictionResult:
        """Make prediction with this model"""
        if not self.is_loaded:
            self.load()
        
        # Update last used time
        import time
        self.last_used = time.time()
        
        # Make prediction based on framework
        if self.metadata.variant_config.framework == "pytorch":
            return self._predict_pytorch(input_data)
        else:
            return self._predict_tensorflow(input_data)
    
    def _predict_pytorch(self, input_data: Any) -> PredictionResult:
        """Make PyTorch prediction"""
        # This would need to be implemented based on your PyTorch model interface
        # For now, return a placeholder
        return PredictionResult(
            colors=[0.25, 0.25, 0.25, 0.25],
            curves=[0.2, 0.2, 0.2, 0.2, 0.2],
            dials=[0.0] * 32,
            confidence=0.5,
            model_name=self.metadata.variant_config.name
        )
    
    def _predict_tensorflow(self, input_data: Any) -> PredictionResult:
        """Make TensorFlow prediction"""
        # This would need to be implemented based on your TensorFlow model interface
        # For now, return a placeholder
        return PredictionResult(
            colors=[0.25, 0.25, 0.25, 0.25],
            curves=[0.2, 0.2, 0.2, 0.2, 0.2],
            dials=[0.0] * 32,
            confidence=0.5,
            model_name=self.metadata.variant_config.name
        )
    
    def unload(self):
        """Unload model from memory"""
        self.model = None
        self.is_loaded = False
        
        # Force garbage collection
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

class AdvancedInferenceEngine:
    """Advanced inference engine supporting multiple models"""
    
    def __init__(self, config: MultiModelInferenceConfig):
        self.config = config
        self.model_manager = ModelManager()
        
        # Loaded model instances
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.active_models: List[str] = []
        
        # Data buffers
        self.data_buffer = queue.Queue(maxsize=config.sequence_length * 2)
        
        # Prediction callbacks
        self.prediction_callbacks: List[Callable[[Dict[str, PredictionResult]], None]] = []
        
        # Statistics
        self.prediction_stats: Dict[str, Dict[str, float]] = {}
        
        logging.info("Advanced inference engine initialized")
    
    def get_available_models(self) -> Dict[str, ModelMetadata]:
        """Get all available trained models"""
        return self.model_manager.get_available_models()
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            metadata = self.model_manager.get_model_by_name(model_name)
            if not metadata:
                logging.error(f"Model {model_name} not found in registry")
                return False
            
            # Check if already loaded
            if model_name in self.loaded_models:
                logging.info(f"Model {model_name} already loaded")
                return True
            
            # Check framework availability
            framework = metadata.variant_config.framework
            if framework == "pytorch" and not PYTORCH_AVAILABLE:
                logging.error(f"PyTorch not available for model {model_name}")
                return False
            elif framework == "tensorflow" and not TENSORFLOW_AVAILABLE:
                logging.error(f"TensorFlow not available for model {model_name}")
                return False
            
            # Create and load model instance
            instance = ModelInstance(metadata)
            instance.load()
            
            self.loaded_models[model_name] = instance
            self.active_models.append(model_name)
            
            # Manage memory - unload old models if necessary
            self._manage_model_memory()
            
            print(f"✅ Loaded model: {model_name}")
            print(f"   Framework: {metadata.variant_config.framework}")
            print(f"   Architecture: {metadata.variant_config.architecture}")
            print(f"   Features: {', '.join(metadata.variant_config.input_features)}")
            print(f"   GPU Used: {metadata.gpu_used}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            return False
    
    def load_multiple_models(self, model_names: List[str]) -> List[str]:
        """Load multiple models, returns successfully loaded model names"""
        loaded = []
        for name in model_names:
            if self.load_model(name):
                loaded.append(name)
        return loaded
    
    def load_best_models(self, max_count: Optional[int] = None) -> List[str]:
        """Load the best performing models"""
        if max_count is None:
            max_count = self.config.max_models
        
        available = self.model_manager.get_available_models()
        if not available:
            logging.warning("No trained models available")
            return []
        
        # Sort by validation loss
        sorted_models = sorted(available.values(), key=lambda x: x.final_val_loss)
        best_models = sorted_models[:max_count]
        
        model_names = [model.variant_config.name for model in best_models]
        return self.load_multiple_models(model_names)
    
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].unload()
            del self.loaded_models[model_name]
            if model_name in self.active_models:
                self.active_models.remove(model_name)
            logging.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all models"""
        for name in list(self.loaded_models.keys()):
            self.unload_model(name)
    
    def _manage_model_memory(self):
        """Manage model memory by unloading old models if necessary"""
        if len(self.loaded_models) > self.config.max_models:
            # Find least recently used models
            lru_models = sorted(
                self.loaded_models.items(),
                key=lambda x: x[1].last_used or 0
            )
            
            # Unload oldest models
            to_unload = len(self.loaded_models) - self.config.max_models
            for i in range(to_unload):
                model_name = lru_models[i][0]
                self.unload_model(model_name)
                logging.info(f"Auto-unloaded LRU model: {model_name}")
    
    def predict_single_model(self, model_name: str, input_data: Any) -> Optional[PredictionResult]:
        """Make prediction with a single model"""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return None
        
        try:
            return self.loaded_models[model_name].predict(input_data)
        except Exception as e:
            logging.error(f"Prediction error with model {model_name}: {e}")
            return None
    
    def predict_all_models(self, input_data: Any) -> Dict[str, PredictionResult]:
        """Make predictions with all loaded models"""
        results = {}
        
        for model_name in self.active_models:
            result = self.predict_single_model(model_name, input_data)
            if result:
                results[model_name] = result
        
        return results
    
    def get_ensemble_prediction(self, input_data: Any) -> Optional[PredictionResult]:
        """Get ensemble prediction from all loaded models"""
        predictions = self.predict_all_models(input_data)
        
        if not predictions:
            return None
        
        # Simple ensemble averaging
        ensemble_colors = [0.0] * 4
        ensemble_curves = [0.0] * 5
        ensemble_dials = [0.0] * 32
        total_confidence = 0.0
        
        for result in predictions.values():
            for i in range(4):
                ensemble_colors[i] += result.colors[i]
            for i in range(5):
                ensemble_curves[i] += result.curves[i]
            for i in range(32):
                ensemble_dials[i] += result.dials[i]
            total_confidence += result.confidence
        
        count = len(predictions)
        ensemble_colors = [x / count for x in ensemble_colors]
        ensemble_curves = [x / count for x in ensemble_curves]
        ensemble_dials = [x / count for x in ensemble_dials]
        avg_confidence = total_confidence / count
        
        return PredictionResult(
            colors=ensemble_colors,
            curves=ensemble_curves,
            dials=ensemble_dials,
            confidence=avg_confidence,
            model_name="ensemble"
        )
    
    def get_model_comparison(self, input_data: Any) -> Dict[str, Any]:
        """Get detailed comparison of all model predictions"""
        predictions = self.predict_all_models(input_data)
        
        if not predictions:
            return {}
        
        comparison = {
            "individual_predictions": predictions,
            "model_count": len(predictions),
            "confidence_stats": {
                "max": max(p.confidence for p in predictions.values()),
                "min": min(p.confidence for p in predictions.values()),
                "avg": sum(p.confidence for p in predictions.values()) / len(predictions)
            }
        }
        
        # Add ensemble prediction
        ensemble = self.get_ensemble_prediction(input_data)
        if ensemble:
            comparison["ensemble_prediction"] = ensemble
        
        return comparison
    
    def get_loaded_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models"""
        info = {}
        
        for name, instance in self.loaded_models.items():
            metadata = instance.metadata
            info[name] = {
                "framework": metadata.variant_config.framework,
                "architecture": metadata.variant_config.architecture,
                "input_features": metadata.variant_config.input_features,
                "hidden_size": metadata.variant_config.hidden_size,
                "final_val_loss": metadata.final_val_loss,
                "gpu_used": metadata.gpu_used,
                "is_loaded": instance.is_loaded,
                "last_used": instance.last_used
            }
        
        return info
    
    def print_status(self):
        """Print current engine status"""
        print(f"\n🧠 =" * 40)
        print(f"  ADVANCED INFERENCE ENGINE STATUS")
        print("=" * 80)
        print(f"📊 Loaded Models: {len(self.loaded_models)}")
        print(f"🔧 Max Models: {self.config.max_models}")
        print(f"🎮 GPU Enabled: {self.config.enable_gpu}")
        
        if self.loaded_models:
            print("\n📋 Loaded Model Details:")
            for name, info in self.get_loaded_models_info().items():
                print(f"   • {name}")
                print(f"     Framework: {info['framework']}, Architecture: {info['architecture']}")
                print(f"     Features: {', '.join(info['input_features'])}")
                print(f"     Val Loss: {info['final_val_loss']:.6f}, GPU: {info['gpu_used']}")
        
        print("=" * 80)