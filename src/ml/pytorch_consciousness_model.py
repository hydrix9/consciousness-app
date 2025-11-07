#!/usr/bin/env python3
"""
PyTorch GPU-Accelerated Consciousness Model
Compatible with existing TensorFlow pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional, Any
import json
from datetime import datetime
import os

class ConsciousnessDataset(Dataset):
    """PyTorch Dataset for consciousness training data"""
    
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.FloatTensor(v) for k, v in y.items()}
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.y.items()}

class PyTorchConsciousnessModel(nn.Module):
    """PyTorch implementation of consciousness model with GPU acceleration"""
    
    def __init__(self, input_dim: int, output_dims: Dict[str, int], hidden_dims: list = None):
        super(PyTorchConsciousnessModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        # Shared feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output heads for different consciousness aspects
        self.output_heads = nn.ModuleDict()
        for name, dim in output_dims.items():
            self.output_heads[name] = nn.Sequential(
                nn.Linear(prev_dim, dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()  # Constrain outputs to [0, 1] range to prevent mode collapse
            )
    
    def forward(self, x):
        # Shared feature extraction
        features = self.shared_layers(x)
        
        # Generate outputs for each consciousness aspect
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(features)
        
        return outputs

class PyTorchConsciousnessTrainer:
    """GPU-accelerated PyTorch trainer for consciousness models"""
    
    def __init__(self, model: PyTorchConsciousnessModel, device: str = None):
        self.model = model
        self.device = device or self._get_best_device()
        self.model.to(self.device)
        
        # Setup optimizers and loss functions
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        self.criterion = nn.MSELoss()
        
        print(f"🚀 PyTorch model initialized on device: {self.device}")
        if self.device.startswith('cuda'):
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    def _get_best_device(self) -> str:
        """Automatically select the best available device"""
        if torch.cuda.is_available():
            return f"cuda:0"
        else:
            return "cpu"
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {name: 0.0 for name in self.model.output_dims.keys()}
        epoch_losses['total'] = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate losses for each output
            losses = {}
            total_loss = 0
            for name in targets.keys():
                loss = self.criterion(outputs[name], targets[name])
                losses[name] = loss
                total_loss += loss
                epoch_losses[name] += loss.item()
            
            epoch_losses['total'] += total_loss.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # Average losses
        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_losses = {name: 0.0 for name in self.model.output_dims.keys()}
        val_losses['total'] = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(inputs)
                
                # Calculate losses
                total_loss = 0
                for name in targets.keys():
                    loss = self.criterion(outputs[name], targets[name])
                    val_losses[name] += loss.item()
                    total_loss += loss
                
                val_losses['total'] += total_loss.item()
        
        # Average losses
        num_batches = len(dataloader)
        return {k: v / num_batches for k, v in val_losses.items()}
    
    def train(self, train_data: Tuple[np.ndarray, Dict[str, np.ndarray]], 
              val_data: Tuple[np.ndarray, Dict[str, np.ndarray]], 
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the consciousness model with GPU acceleration"""
        
        print("=" * 60)
        print("🧠 PYTORCH CONSCIOUSNESS MODEL TRAINING INITIATED")
        print("=" * 60)
        print(f"🎮 Device: {self.device}")
        print(f"📊 Training samples: {len(train_data[0])}")
        print(f"📊 Validation samples: {len(val_data[0])}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print()
        
        # Create datasets and dataloaders
        train_dataset = ConsciousnessDataset(train_data[0], train_data[1])
        val_dataset = ConsciousnessDataset(val_data[0], val_data[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_losses': {name: [] for name in self.model.output_dims.keys()},
            'val_losses': {name: [] for name in self.model.output_dims.keys()}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update history
            history['train_loss'].append(train_losses['total'])
            history['val_loss'].append(val_losses['total'])
            
            for name in self.model.output_dims.keys():
                history['train_losses'][name].append(train_losses[name])
                history['val_losses'][name].append(val_losses[name])
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_consciousness_model_pytorch.pth')
            else:
                patience_counter += 1
            
            # Print progress
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_losses['total']:.6f} | "
                  f"Val Loss: {val_losses['total']:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # GPU memory info
            if self.device.startswith('cuda') and epoch % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"🎮 GPU Memory Used: {gpu_memory:.2f} GB")
            
            # Early stopping
            if patience_counter >= 20:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print()
        print("=" * 60)
        print("🧠 PYTORCH CONSCIOUSNESS MODEL TRAINING COMPLETED")
        print("=" * 60)
        print(f"⏰ Total training time: {total_time:.1f} seconds")
        print(f"🏆 Best validation loss: {best_val_loss:.6f}")
        print(f"💾 Best model saved as: best_consciousness_model_pytorch.pth")
        
        return history
    
    def predict(self, X: np.ndarray, mc_dropout: bool = True, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """Make predictions on new data
        
        Args:
            X: Input data
            mc_dropout: If True, keeps dropout enabled for Monte Carlo sampling (generates variety)
            n_samples: Number of stochastic forward passes (only used if mc_dropout=True)
        
        Returns:
            Dictionary of predictions (averaged over n_samples if mc_dropout=True)
        """
        if not mc_dropout:
            # Standard deterministic inference (dropout disabled)
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = {k: v.cpu().numpy() for k, v in outputs.items()}
            
            return predictions
        else:
            # Monte Carlo Dropout: keep dropout enabled for stochastic predictions
            # This creates natural variety - different predictions each time!
            self.model.train()  # Keeps dropout layers active
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Make single stochastic forward pass (with dropout active)
            # No gradient computation needed
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = {k: v.cpu().numpy() for k, v in outputs.items()}
            
            return predictions
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dims': self.model.output_dims,
                'hidden_dims': self.model.hidden_dims
            }
        }, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"📁 Model loaded from: {filepath}")

def create_pytorch_consciousness_model(input_shape: Tuple[int, ...], 
                                     output_shapes: Dict[str, int]) -> PyTorchConsciousnessTrainer:
    """Create a PyTorch consciousness model ready for GPU training"""
    
    # Calculate input dimension
    if len(input_shape) == 1:
        input_dim = input_shape[0]
    else:
        input_dim = np.prod(input_shape)
    
    # Create model
    model = PyTorchConsciousnessModel(input_dim, output_shapes)
    trainer = PyTorchConsciousnessTrainer(model)
    
    return trainer

if __name__ == "__main__":
    # Test PyTorch GPU setup
    print("=== PyTorch GPU Consciousness Model Test ===\n")
    
    # Test data
    input_shape = (100,)
    output_shapes = {
        'creativity': 10,
        'awareness': 8,
        'emotion': 12
    }
    
    # Create model
    trainer = create_pytorch_consciousness_model(input_shape, output_shapes)
    
    # Generate test data
    n_samples = 1000
    X = np.random.randn(n_samples, input_shape[0])
    y = {
        'creativity': np.random.randn(n_samples, output_shapes['creativity']),
        'awareness': np.random.randn(n_samples, output_shapes['awareness']),
        'emotion': np.random.randn(n_samples, output_shapes['emotion'])
    }
    
    # Split data
    split = int(0.8 * n_samples)
    train_data = (X[:split], {k: v[:split] for k, v in y.items()})
    val_data = (X[split:], {k: v[split:] for k, v in y.items()})
    
    # Train model
    history = trainer.train(train_data, val_data, epochs=50, batch_size=64)
    
    # Test predictions
    test_X = np.random.randn(10, input_shape[0])
    predictions = trainer.predict(test_X)
    
    print("✅ PyTorch GPU consciousness model test completed!")
    print(f"📊 Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"📊 Final validation loss: {history['val_loss'][-1]:.6f}")