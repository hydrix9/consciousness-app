"""
Advanced Model Manager for Consciousness Training
Supports multiple model variants and GPU acceleration
"""

import json
import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime

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
class ModelVariantConfig:
    """Configuration for a specific model variant"""
    name: str
    framework: str  # 'pytorch' or 'tensorflow'
    architecture: str  # 'lstm', 'gru', 'transformer', 'cnn_lstm', etc.
    input_features: List[str]  # ['rng', 'eeg', 'combined', etc.]
    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.3
    sequence_length: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    max_epochs: int = 50
    use_gpu: bool = True
    description: str = ""
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    variant_config: ModelVariantConfig
    training_time: str
    final_loss: float
    final_val_loss: float
    total_epochs: int
    training_samples: int
    model_path: str
    framework_version: str
    gpu_used: bool
    performance_metrics: Dict[str, float]
    
    def to_dict(self):
        return {
            'variant_config': self.variant_config.to_dict(),
            'training_time': self.training_time,
            'final_loss': self.final_loss,
            'final_val_loss': self.final_val_loss,
            'total_epochs': self.total_epochs,
            'training_samples': self.training_samples,
            'model_path': self.model_path,
            'framework_version': self.framework_version,
            'gpu_used': self.gpu_used,
            'performance_metrics': self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            variant_config=ModelVariantConfig.from_dict(data['variant_config']),
            **{k: v for k, v in data.items() if k != 'variant_config'}
        )

class ModelManager:
    """Advanced model management for consciousness training"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.registry_file = self.models_dir / "model_registry.json"
        self.models_registry: Dict[str, ModelMetadata] = {}
        self.load_registry()
        
        # Available frameworks
        self.frameworks = {
            'pytorch': PYTORCH_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE
        }
        
        logging.info(f"Model Manager initialized. Available frameworks: {[k for k, v in self.frameworks.items() if v]}")
    
    def get_default_variants(self) -> List[ModelVariantConfig]:
        """Get default model variants to train"""
        variants = []
        
        # Basic RNG-only models
        variants.append(ModelVariantConfig(
            name="rng_lstm_basic",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="lstm",
            input_features=["rng"],
            hidden_size=64,
            description="Basic LSTM using only RNG data"
        ))
        
        variants.append(ModelVariantConfig(
            name="rng_gru_basic",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="gru",
            input_features=["rng"],
            hidden_size=64,
            description="GRU variant using only RNG data"
        ))
        
        # EEG-only models
        variants.append(ModelVariantConfig(
            name="eeg_lstm_basic",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="lstm",
            input_features=["eeg"],
            hidden_size=128,
            description="LSTM using only EEG data"
        ))
        
        # Combined models
        variants.append(ModelVariantConfig(
            name="combined_lstm_standard",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="lstm",
            input_features=["rng", "eeg"],
            hidden_size=128,
            num_layers=3,
            description="Standard LSTM using RNG + EEG data"
        ))
        
        variants.append(ModelVariantConfig(
            name="combined_transformer",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="transformer",
            input_features=["rng", "eeg"],
            hidden_size=256,
            description="Transformer architecture for RNG + EEG"
        ))
        
        # Deep variants
        variants.append(ModelVariantConfig(
            name="rng_deep_lstm",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="lstm",
            input_features=["rng"],
            hidden_size=128,
            num_layers=4,
            description="Deep LSTM with 4 layers for RNG"
        ))
        
        variants.append(ModelVariantConfig(
            name="combined_cnn_lstm",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="cnn_lstm",
            input_features=["rng", "eeg"],
            hidden_size=128,
            description="CNN-LSTM hybrid for temporal patterns"
        ))
        
        # Experimental variants
        variants.append(ModelVariantConfig(
            name="rng_lightweight",
            framework="pytorch" if PYTORCH_AVAILABLE else "tensorflow",
            architecture="lstm",
            input_features=["rng"],
            hidden_size=32,
            num_layers=1,
            description="Lightweight model for fast inference"
        ))
        
        return variants
    
    def register_model(self, model_metadata: ModelMetadata):
        """Register a trained model"""
        self.models_registry[model_metadata.variant_config.name] = model_metadata
        self.save_registry()
        logging.info(f"Registered model: {model_metadata.variant_config.name}")
    
    def get_available_models(self) -> Dict[str, ModelMetadata]:
        """Get all available trained models"""
        return self.models_registry.copy()
    
    def get_model_by_name(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name"""
        return self.models_registry.get(name)
    
    def get_models_by_framework(self, framework: str) -> List[ModelMetadata]:
        """Get models by framework"""
        return [model for model in self.models_registry.values() 
                if model.variant_config.framework == framework]
    
    def get_models_by_features(self, features: List[str]) -> List[ModelMetadata]:
        """Get models by input features"""
        return [model for model in self.models_registry.values() 
                if set(model.variant_config.input_features) == set(features)]
    
    def get_best_model(self, metric: str = "final_val_loss", 
                      features: Optional[List[str]] = None) -> Optional[ModelMetadata]:
        """Get the best model by a specific metric"""
        candidates = self.models_registry.values()
        
        if features:
            candidates = [model for model in candidates 
                         if set(model.variant_config.input_features) == set(features)]
        
        if not candidates:
            return None
        
        if metric == "final_val_loss":
            return min(candidates, key=lambda x: x.final_val_loss)
        elif metric == "final_loss":
            return min(candidates, key=lambda x: x.final_loss)
        else:
            # Look in performance metrics
            valid_candidates = [model for model in candidates 
                              if metric in model.performance_metrics]
            if valid_candidates:
                if metric.endswith("_loss") or metric.endswith("_error"):
                    return min(valid_candidates, key=lambda x: x.performance_metrics[metric])
                else:
                    return max(valid_candidates, key=lambda x: x.performance_metrics[metric])
        
        return None
    
    def load_registry(self):
        """Load model registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.models_registry = {
                        name: ModelMetadata.from_dict(metadata_dict)
                        for name, metadata_dict in data.items()
                    }
                logging.info(f"Loaded {len(self.models_registry)} models from registry")
            except Exception as e:
                logging.error(f"Error loading model registry: {e}")
                self.models_registry = {}
        else:
            self.models_registry = {}
    
    def save_registry(self):
        """Save model registry to file"""
        try:
            data = {
                name: metadata.to_dict()
                for name, metadata in self.models_registry.items()
            }
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving model registry: {e}")
    
    def create_model_path(self, variant_config: ModelVariantConfig) -> str:
        """Create a unique model path and ensure directory exists"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{variant_config.name}_{timestamp}"
        model_path = self.models_dir / filename
        
        # Create model directory
        model_path.mkdir(parents=True, exist_ok=True)
        
        return str(model_path)
    
    def save_model_files(self, model, variant_config: ModelVariantConfig, model_path: str) -> bool:
        """Save model weights and configuration files"""
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model weights based on framework
            if variant_config.framework == "pytorch":
                import torch
                weights_path = model_dir / "model.pth"
                torch.save(model.state_dict(), weights_path)
                logging.info(f"Saved PyTorch weights to {weights_path}")
                
            elif variant_config.framework == "tensorflow":
                weights_path = model_dir / "model.h5"
                model.save(weights_path)
                logging.info(f"Saved TensorFlow model to {weights_path}")
            
            # Save model configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                import json
                config_dict = {
                    "name": variant_config.name,
                    "framework": variant_config.framework,
                    "architecture": variant_config.architecture,
                    "input_features": variant_config.input_features,
                    "hidden_size": variant_config.hidden_size,
                    "num_layers": variant_config.num_layers,
                    "dropout_rate": variant_config.dropout_rate,
                    "sequence_length": variant_config.sequence_length,
                    "batch_size": variant_config.batch_size,
                    "learning_rate": variant_config.learning_rate,
                    "max_epochs": variant_config.max_epochs,
                    "use_gpu": variant_config.use_gpu,
                    "description": variant_config.description
                }
                json.dump(config_dict, f, indent=2)
            
            logging.info(f"Saved model configuration to {config_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model files: {e}")
            return False
    
    def delete_model(self, name: str) -> bool:
        """Delete a model and its files"""
        if name not in self.models_registry:
            return False
        
        try:
            metadata = self.models_registry[name]
            model_path = Path(metadata.model_path)
            
            # Delete model files
            for ext in ['.h5', '.pth', '.json', '_metadata.json']:
                file_path = model_path.with_suffix(ext)
                if file_path.exists():
                    file_path.unlink()
            
            # Remove from registry
            del self.models_registry[name]
            self.save_registry()
            
            logging.info(f"Deleted model: {name}")
            return True
        except Exception as e:
            logging.error(f"Error deleting model {name}: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary statistics of available models"""
        if not self.models_registry:
            return {"total_models": 0}
        
        frameworks = {}
        architectures = {}
        features = {}
        
        for model in self.models_registry.values():
            # Count frameworks
            fw = model.variant_config.framework
            frameworks[fw] = frameworks.get(fw, 0) + 1
            
            # Count architectures
            arch = model.variant_config.architecture
            architectures[arch] = architectures.get(arch, 0) + 1
            
            # Count feature combinations
            feat_key = "_".join(sorted(model.variant_config.input_features))
            features[feat_key] = features.get(feat_key, 0) + 1
        
        return {
            "total_models": len(self.models_registry),
            "frameworks": frameworks,
            "architectures": architectures,
            "feature_combinations": features,
            "available_frameworks": [k for k, v in self.frameworks.items() if v]
        }
    
    def print_model_summary(self):
        """Print a formatted summary of available models"""
        summary = self.get_model_summary()
        
        print("\n🧠 =" * 40)
        print("  CONSCIOUSNESS MODEL REGISTRY SUMMARY")
        print("=" * 80)
        print(f"📊 Total Models: {summary['total_models']}")
        
        if summary['total_models'] > 0:
            print(f"🔧 Frameworks: {', '.join(f'{k}: {v}' for k, v in summary['frameworks'].items())}")
            print(f"🏗️  Architectures: {', '.join(f'{k}: {v}' for k, v in summary['architectures'].items())}")
            print(f"📈 Feature Sets: {', '.join(f'{k}: {v}' for k, v in summary['feature_combinations'].items())}")
        
        available_frameworks = summary.get('available_frameworks', [])
        print(f"💻 Available Frameworks: {', '.join(available_frameworks)}")
        print("=" * 80)