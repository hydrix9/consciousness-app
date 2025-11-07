"""
Multi-Model Training System for Consciousness Models
Supports training multiple variants with different configurations
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .model_manager import ModelManager, ModelVariantConfig, ModelMetadata

try:
    from .training_pipeline import TrainingConfig
except ImportError:
    # Fallback if training_pipeline is not available
    class TrainingConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

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

class MultiModelTrainer:
    """Advanced trainer for multiple consciousness model variants"""
    
    def __init__(self, model_manager: ModelManager = None):
        if model_manager is None:
            self.model_manager = ModelManager()
        else:
            self.model_manager = model_manager
        self.training_results: List[Dict[str, Any]] = []

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

class MultiModelTrainer:
    """Advanced trainer for multiple consciousness model variants"""
    
    def __init__(self, model_manager: ModelManager = None):
        if model_manager is None:
            self.model_manager = ModelManager()
        else:
            self.model_manager = model_manager
        self.training_results: List[Dict[str, Any]] = []
        
    def train_variant(self, variant_config: ModelVariantConfig, 
                     data_files: List[str]) -> Optional[ModelMetadata]:
        """Train a single model variant"""
        
        print(f"\n🧠 =" * 40)
        print(f"  TRAINING MODEL VARIANT: {variant_config.name.upper()}")
        print("=" * 80)
        print(f"🔧 Framework: {variant_config.framework.upper()}")
        print(f"🏗️  Architecture: {variant_config.architecture}")
        print(f"📊 Input Features: {', '.join(variant_config.input_features)}")
        print(f"🧠 Hidden Size: {variant_config.hidden_size}")
        print(f"📐 Layers: {variant_config.num_layers}")
        print(f"🎯 Max Epochs: {variant_config.max_epochs}")
        print(f"📦 Batch Size: {variant_config.batch_size}")
        print(f"⚡ Learning Rate: {variant_config.learning_rate}")
        print(f"🎮 GPU Enabled: {variant_config.use_gpu}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Create training config
            training_config = TrainingConfig(
                model_type=variant_config.architecture,
                sequence_length=variant_config.sequence_length,
                batch_size=variant_config.batch_size,
                epochs=variant_config.max_epochs,  # Use 'epochs' instead of 'max_epochs'
                learning_rate=variant_config.learning_rate,
                hidden_size=variant_config.hidden_size
            )
            
            # Determine which trainer to use
            trained_model = None
            if variant_config.framework == "pytorch" and PYTORCH_AVAILABLE:
                result = self._train_pytorch_variant(
                    variant_config, training_config, data_files
                )
                if len(result) == 5:  # New format with model
                    success, final_loss, val_loss, epochs, trained_model = result
                else:  # Old format fallback
                    success, final_loss, val_loss, epochs = result
                framework_version = torch.__version__
                gpu_used = variant_config.use_gpu and torch.cuda.is_available()
            
            elif variant_config.framework == "tensorflow" and TENSORFLOW_AVAILABLE:
                success, final_loss, val_loss, epochs = self._train_tensorflow_variant(
                    variant_config, training_config, data_files
                )
                framework_version = tf.__version__
                gpu_used = variant_config.use_gpu and len(tf.config.list_physical_devices('GPU')) > 0
            
            else:
                print(f"❌ Framework {variant_config.framework} not available!")
                return None
            
            if not success:
                print(f"❌ Training failed for {variant_config.name}")
                return None
            
            training_time = time.time() - start_time
            
            # Create model path
            model_path = self.model_manager.create_model_path(variant_config)
            
            # Save model files if we have a trained model
            if trained_model is not None:
                save_success = self.model_manager.save_model_files(trained_model, variant_config, model_path)
                if save_success:
                    print(f"💾 Model saved successfully to {model_path}")
                else:
                    print(f"⚠️ Warning: Model training succeeded but saving failed")
            
            # Create metadata
            metadata = ModelMetadata(
                variant_config=variant_config,
                training_time=datetime.now().isoformat(),
                final_loss=final_loss,
                final_val_loss=val_loss,
                total_epochs=epochs,
                training_samples=len(data_files),  # This could be more accurate
                model_path=model_path,
                framework_version=framework_version,
                gpu_used=gpu_used,
                performance_metrics={
                    "training_time_seconds": training_time,
                    "epochs_per_second": epochs / training_time if training_time > 0 else 0
                }
            )
            
            # Register the model
            self.model_manager.register_model(metadata)
            
            print(f"✅ Successfully trained {variant_config.name}")
            print(f"🏆 Final Loss: {final_loss:.6f}")
            print(f"🏆 Validation Loss: {val_loss:.6f}")
            print(f"⏰ Training Time: {training_time:.1f}s")
            print(f"📁 Saved to: {model_path}")
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error training {variant_config.name}: {e}")
            logging.error(f"Error training {variant_config.name}: {e}")
            return None
    
    def _train_pytorch_variant(self, variant_config: ModelVariantConfig, 
                              training_config: TrainingConfig, 
                              data_files: List[str]) -> tuple:
        """Train a PyTorch variant with GPU acceleration"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            import numpy as np
            
            # Configure GPU/CUDA device
            if variant_config.use_gpu and torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"🎮 GPU ACCELERATION ENABLED!")
                print(f"   Device: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                
                # Enable cuDNN benchmarking for optimal performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Clear GPU cache before training
                torch.cuda.empty_cache()
            else:
                device = torch.device('cpu')
                if variant_config.use_gpu:
                    print(f"⚠️  GPU requested but not available, using CPU")
                else:
                    print(f"💻 CPU mode selected")
            
            print(f"📱 Device: {device}")
            print(f"⚡ Training {variant_config.name}")
            print(f"🎯 Architecture: {variant_config.architecture}")
            print(f"📊 Features: {', '.join(variant_config.input_features)}")
            
            # Create model based on architecture
            model = self._create_pytorch_model(variant_config, device)
            
            # Generate synthetic training data if no real data available
            if not data_files or len(data_files) == 0:
                print("� Using synthetic training data (no real data files found)")
                X_train, y_train, X_val, y_val = self._generate_synthetic_data(variant_config)
            else:
                print(f"📁 Loading data from {len(data_files)} files")
                X_train, y_train, X_val, y_val = self._load_real_data(data_files, variant_config)
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            X_val = torch.FloatTensor(X_val).to(device)
            y_val = torch.FloatTensor(y_val).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=variant_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=variant_config.batch_size)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=variant_config.learning_rate)
            
            # Training loop
            print("🔄 Training in progress...")
            best_val_loss = float('inf')
            epochs_trained = 0
            
            for epoch in range(variant_config.max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                epochs_trained = epoch + 1
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                elif epoch > 10 and val_loss > best_val_loss * 1.1:  # Allow some tolerance
                    print(f"🛑 Early stopping at epoch {epochs_trained}")
                    break
                
                if epoch % 10 == 0:
                    print(f"📊 Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            final_loss = train_loss
            final_val_loss = best_val_loss
            
            print(f"✅ Training completed successfully!")
            print(f"🏆 Final Loss: {final_loss:.6f}")
            print(f"🏆 Validation Loss: {final_val_loss:.6f}")
            print(f"📊 Epochs: {epochs_trained}")
            
            # Return the trained model along with metrics
            return True, final_loss, final_val_loss, epochs_trained, model
                
        except Exception as e:
            logging.error(f"PyTorch training error: {e}")
            return False, 0.0, 0.0, 0, None
        except Exception as e:
            logging.error(f"PyTorch training error: {e}")
            return False, 0.0, 0.0, 0
    
    def _train_tensorflow_variant(self, variant_config: ModelVariantConfig,
                                 training_config: TrainingConfig,
                                 data_files: List[str]) -> tuple:
        """Train a TensorFlow variant"""
        try:
            from .training_pipeline import ModelTrainer
            
            # Create specialized trainer
            trainer = ModelTrainer()
            
            # Train model with specific architecture
            if "rng" in variant_config.input_features and "eeg" not in variant_config.input_features:
                # RNG only
                trainer.train_mode1(data_files, training_config)
                final_loss = getattr(trainer.mode1, 'final_loss', 0.0)
                val_loss = getattr(trainer.mode1, 'final_val_loss', 0.0)
                epochs = getattr(trainer.mode1, 'total_epochs', 0)
            elif "eeg" in variant_config.input_features:
                # EEG or combined
                trainer.train_mode2(data_files, training_config)
                final_loss = getattr(trainer.mode2, 'final_loss', 0.0)
                val_loss = getattr(trainer.mode2, 'final_val_loss', 0.0)
                epochs = getattr(trainer.mode2, 'total_epochs', 0)
            else:
                raise ValueError(f"Unsupported feature combination: {variant_config.input_features}")
            
            # Save the model with variant name
            model_path = self.model_manager.create_model_path(variant_config)
            if "rng" in variant_config.input_features and "eeg" not in variant_config.input_features:
                trainer.mode1.save_model(model_path)
            else:
                trainer.mode2.save_model(model_path)
            
            return True, final_loss, val_loss, epochs
            
        except Exception as e:
            logging.error(f"TensorFlow training error: {e}")
            return False, 0.0, 0.0, 0
    
    def train_multiple_variants(self, variants: List[Union[str, ModelVariantConfig]], 
                              data_files: List[str]) -> List[ModelMetadata]:
        """Train multiple model variants"""
        
        print(f"\n🚀 =" * 40)
        print(f"  MULTI-MODEL CONSCIOUSNESS TRAINING SESSION")
        print("=" * 80)
        print(f"📊 Training {len(variants)} model variants")
        print(f"📁 Using {len(data_files)} data files")
        print("=" * 80)
        
        trained_models = []
        
        # Convert string names to ModelVariantConfig objects if needed
        variant_configs = []
        for variant in variants:
            if isinstance(variant, str):
                # Find variant by name from default variants
                default_variants = self.model_manager.get_default_variants()
                variant_config = next((v for v in default_variants if v.name == variant), None)
                if variant_config is None:
                    print(f"❌ Unknown variant name: {variant}")
                    continue
                variant_configs.append(variant_config)
            else:
                variant_configs.append(variant)
        
        for i, variant in enumerate(variant_configs, 1):
            print(f"\n🔄 Training variant {i}/{len(variant_configs)}: {variant.name}")
            
            metadata = self.train_variant(variant, data_files)
            if metadata:
                trained_models.append(metadata)
            
            print(f"📈 Progress: {i}/{len(variants)} variants completed")
        
        print(f"\n🎉 =" * 40)
        print(f"  MULTI-MODEL TRAINING COMPLETED")
        print("=" * 80)
        print(f"✅ Successfully trained: {len(trained_models)}/{len(variants)} models")
        
        if trained_models:
            # Show summary of best models
            best_overall = min(trained_models, key=lambda x: x.final_val_loss)
            print(f"🏆 Best Overall Model: {best_overall.variant_config.name}")
            print(f"   Validation Loss: {best_overall.final_val_loss:.6f}")
            print(f"   Framework: {best_overall.variant_config.framework}")
            print(f"   Architecture: {best_overall.variant_config.architecture}")
        
        self.model_manager.print_model_summary()
        
        return trained_models
    
    def train_default_variants(self, data_files: List[str]) -> List[ModelMetadata]:
        """Train all default model variants"""
        variants = self.model_manager.get_default_variants()
        return self.train_multiple_variants(variants, data_files)
    
    def create_custom_variant(self, name: str, framework: str, architecture: str,
                            input_features: List[str], **kwargs) -> ModelVariantConfig:
        """Create a custom model variant configuration"""
        return ModelVariantConfig(
            name=name,
            framework=framework,
            architecture=architecture,
            input_features=input_features,
            **kwargs
        )
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks"""
        available = []
        if PYTORCH_AVAILABLE:
            available.append("pytorch")
        if TENSORFLOW_AVAILABLE:
            available.append("tensorflow")
        return available
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of the current training session"""
        return {
            "total_variants_trained": len(self.training_results),
            "available_frameworks": self.get_available_frameworks(),
            "model_manager_summary": self.model_manager.get_model_summary()
        }
    
    def _create_pytorch_model(self, variant_config: ModelVariantConfig, device: str):
        """Create a PyTorch model based on variant configuration"""
        import torch
        import torch.nn as nn
        
        # Determine input size based on features
        input_size = 0
        if "rng" in variant_config.input_features:
            input_size += 8  # RNG features
        if "eeg" in variant_config.input_features:
            input_size += 32  # EEG features
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  dropout=dropout_rate, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()  # Constrain output to [0, 1]
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])  # Use last output
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
                out = self.fc(out[:, -1, :])  # Use last output
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
                # x shape: (batch, seq_len, features)
                x = x.transpose(1, 2)  # (batch, features, seq_len)
                x = torch.relu(self.conv1d(x))
                x = x.transpose(1, 2)  # Back to (batch, seq_len, features)
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)  # Apply sigmoid activation
                return out
        
        # Create model based on architecture
        if variant_config.architecture == "lstm":
            model = LSTMModel(input_size, variant_config.hidden_size, 
                            variant_config.num_layers, variant_config.dropout_rate)
        elif variant_config.architecture == "gru":
            model = GRUModel(input_size, variant_config.hidden_size,
                           variant_config.num_layers, variant_config.dropout_rate)
        elif variant_config.architecture == "transformer":
            model = TransformerModel(input_size, variant_config.hidden_size,
                                   variant_config.num_layers, variant_config.dropout_rate)
        elif variant_config.architecture == "cnn_lstm":
            model = CNNLSTMModel(input_size, variant_config.hidden_size,
                               variant_config.num_layers, variant_config.dropout_rate)
        else:
            raise ValueError(f"Unsupported architecture: {variant_config.architecture}")
        
        return model.to(device)
    
    def _generate_synthetic_data(self, variant_config: ModelVariantConfig):
        """Generate synthetic data for training when no real data is available"""
        import numpy as np
        
        # Determine input size based on features
        input_size = 0
        if "rng" in variant_config.input_features:
            input_size += 8  # RNG features
        if "eeg" in variant_config.input_features:
            input_size += 32  # EEG features
        
        # Generate synthetic sequential data
        n_train_samples = 1000
        n_val_samples = 200
        seq_len = variant_config.sequence_length
        
        # Training data
        X_train = np.random.randn(n_train_samples, seq_len, input_size)
        y_train = np.random.randn(n_train_samples, 1)
        
        # Validation data
        X_val = np.random.randn(n_val_samples, seq_len, input_size)
        y_val = np.random.randn(n_val_samples, 1)
        
        return X_train, y_train, X_val, y_val
    
    def _load_real_data(self, data_files: List[str], variant_config: ModelVariantConfig):
        """Load real data from session files"""
        import numpy as np
        
        try:
            # Import the real data loader
            try:
                from data.real_data_loader import RealDataLoader
            except ImportError:
                from src.data.real_data_loader import RealDataLoader
            
            print(f"📁 Loading real data from {len(data_files)} session files...")
            
            # Initialize the data loader
            data_loader = RealDataLoader()
            
            # Load all sessions
            sessions = data_loader.load_multiple_sessions(data_files)
            
            if not sessions:
                print("⚠️  No valid sessions loaded, falling back to synthetic data")
                return self._generate_synthetic_data(variant_config)
            
            print(f"✅ Loaded {len(sessions)} sessions successfully")
            
            # Prepare training data from sessions
            training_data = data_loader.prepare_training_data(
                sessions, 
                sequence_length=variant_config.sequence_length,
                prediction_horizon=1
            )
            
            if not training_data:
                print("⚠️  No training data prepared, falling back to synthetic data")
                return self._generate_synthetic_data(variant_config)
            
            # Extract features based on variant config
            # The RealDataLoader returns pre-sequenced data with keys like:
            # 'rng_inputs', 'eeg_inputs', 'combined_inputs' for inputs
            # 'color_targets', 'position_targets', etc. for targets
            
            input_sequences = []
            
            if "rng" in variant_config.input_features:
                if 'rng_inputs' in training_data:
                    rng_inputs = training_data['rng_inputs']
                    input_sequences.append(rng_inputs)
                    print(f"  🎲 RNG inputs: {rng_inputs.shape}")
                else:
                    print("  ⚠️  RNG inputs requested but not available")
            
            if "eeg" in variant_config.input_features:
                if 'eeg_inputs' in training_data:
                    eeg_inputs = training_data['eeg_inputs']
                    input_sequences.append(eeg_inputs)
                    print(f"  🧠 EEG inputs: {eeg_inputs.shape}")
                else:
                    print("  ⚠️  EEG inputs requested but not available")
            
            # Check for combined inputs (RNG + EEG)
            if len(input_sequences) == 0 and 'combined_inputs' in training_data:
                input_sequences.append(training_data['combined_inputs'])
                print(f"  🔄 Using combined RNG+EEG inputs: {training_data['combined_inputs'].shape}")
            
            if not input_sequences:
                print("⚠️  No input sequences available, falling back to synthetic data")
                return self._generate_synthetic_data(variant_config)
            
            # Combine input sequences if multiple
            if len(input_sequences) > 1:
                X_sequences = np.concatenate(input_sequences, axis=2)  # Concatenate along feature dimension
                print(f"  📊 Combined inputs: {X_sequences.shape}")
            else:
                X_sequences = input_sequences[0]
                print(f"  📊 Using inputs: {X_sequences.shape}")
            
            # Extract targets (predictions)
            # Priority: color_targets > position_targets > consciousness_targets > dimension_targets
            target_key = None
            for key in ['color_targets', 'position_targets', 'consciousness_targets', 'dimension_targets']:
                if key in training_data:
                    target_key = key
                    break
            
            if target_key:
                y_sequences = training_data[target_key]
                print(f"  🎯 Using {target_key}: {y_sequences.shape}")
                
                # Ensure y_sequences matches X_sequences length
                if len(y_sequences) != len(X_sequences):
                    min_len = min(len(y_sequences), len(X_sequences))
                    y_sequences = y_sequences[:min_len]
                    X_sequences = X_sequences[:min_len]
                    print(f"  ⚙️  Aligned sequences to {min_len} samples")
                
                # Reshape targets if needed
                if y_sequences.ndim == 1:
                    y_sequences = y_sequences.reshape(-1, 1)
                elif y_sequences.ndim > 2:
                    # Flatten multi-dimensional targets
                    y_sequences = y_sequences.reshape(len(y_sequences), -1)
            else:
                print("  ⚠️  No targets found, using next-step feature prediction")
                # Use next step of inputs as targets
                y_sequences = X_sequences[1:, -1, :]  # Last timestep, all features
                X_sequences = X_sequences[:-1, :, :]   # All but last sample
                y_sequences = y_sequences.reshape(-1, 1) if y_sequences.ndim == 1 else y_sequences
            
            # Split into train/validation (80/20)
            split_idx = int(0.8 * len(X_sequences))
            
            X_train = X_sequences[:split_idx]
            y_train = y_sequences[:split_idx]
            X_val = X_sequences[split_idx:]
            y_val = y_sequences[split_idx:]
            
            print(f"✅ Prepared training data:")
            print(f"  📦 Train: X={X_train.shape}, y={y_train.shape}")
            print(f"  📦 Val:   X={X_val.shape}, y={y_val.shape}")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Falling back to synthetic data")
            return self._generate_synthetic_data(variant_config)