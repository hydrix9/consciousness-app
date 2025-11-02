"""
Main Application Entry Point

This module ties together all components of the consciousness data generation
and machine learning application.
"""

import sys
import os
import argparse
import logging
import yaml
from typing import Optional, Dict, Any, List

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
    from PyQt5.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from hardware.truerng_v3 import TrueRNGV3
from hardware.emotiv_eeg import EmotivEEG, MockEmotivEEG
from hardware.eeg_bridge import EEGBridge, load_eeg_config, EEGSourceType
from gui.painting_interface import create_painting_app
from ml.inference_interface import create_inference_app
from ml.training_pipeline import ModelTrainer, TrainingConfig
from data.data_logger import DataLogger


class ConsciousnessApp:
    """Main application class"""
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """
        Initialize the consciousness application
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # Hardware devices
        self.rng_device: Optional[TrueRNGV3] = None
        self.eeg_bridge: Optional[EEGBridge] = None
        
        # Components
        self.data_logger: Optional[DataLogger] = None
        self.model_trainer: Optional[ModelTrainer] = None
        
        # GUI applications
        self.painting_app = None
        self.inference_app = None
        
        # Setup logging
        self.setup_logging()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            config_file = os.path.join(os.path.dirname(__file__), "..", self.config_path)
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.warning(f"Could not load config file {self.config_path}: {e}")
            return self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'app': {'name': 'Consciousness App', 'debug': False},
            'hardware': {
                'truerng': {'device_path': 'auto', 'baud_rate': 3000000},
                'emotiv': {'client_id': '', 'client_secret': '', 'license': ''}
            },
            'timing': {'drawing_delay_offset': 0.2, 'sampling_rate': 1000},
            'data': {'output_directory': 'data', 'file_format': 'json'},
            'ml': {'model_type': 'lstm', 'sequence_length': 100}
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.config.get('app', {}).get('debug', False) else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('consciousness_app.log')
            ]
        )
        
    def initialize_hardware(self) -> bool:
        """Initialize hardware devices"""
        success = True
        
        # Initialize TrueRNG device
        try:
            rng_config = self.config.get('hardware', {}).get('truerng', {})
            self.rng_device = TrueRNGV3(
                device_path=rng_config.get('device_path', 'auto'),
                baud_rate=rng_config.get('baud_rate', 3000000)
            )
            
            if self.rng_device.connect():
                logging.info("TrueRNG V3 connected successfully")
            else:
                logging.warning("Failed to connect to TrueRNG V3, will continue without it")
                self.rng_device = None
                success = False
                
        except Exception as e:
            logging.error(f"Error initializing TrueRNG: {e}")
            self.rng_device = None
            success = False
            
        # Initialize EEG Bridge
        try:
            # Load EEG configuration
            eeg_config = load_eeg_config()
            
            # Override with config file settings if available
            app_eeg_config = self.config.get('eeg', {})
            if app_eeg_config:
                eeg_config.update(app_eeg_config)
            
            # Create EEG bridge
            self.eeg_bridge = EEGBridge(eeg_config)
            
            # Connect with automatic fallback
            import asyncio
            connected = asyncio.run(self.eeg_bridge.connect())
            
            if connected:
                connection_info = self.eeg_bridge.get_connection_info()
                logging.info(f"EEG connected via {connection_info.source.value}: {connection_info.device_info.get('name', 'Unknown')}")
            else:
                logging.error("Failed to connect to any EEG source")
                success = False
                
        except Exception as e:
            logging.error(f"Error initializing EEG Bridge: {e}")
            self.eeg_bridge = None
            success = False
            
        return success
        
    def find_training_files(self, data_dir: str) -> List[str]:
        """
        Find all training session files in a directory
        
        Args:
            data_dir: Directory to search for training files
            
        Returns:
            List of file paths
        """
        training_files = []
        
        if not os.path.exists(data_dir):
            logging.error(f"Data directory does not exist: {data_dir}")
            return training_files
            
        # Search for supported file formats
        supported_extensions = ['.json', '.h5', '.hdf5']
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file has supported extension
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    # Additional check for session files (contain 'session' in name)
                    if 'session' in file.lower():
                        training_files.append(file_path)
                        logging.debug(f"Found training file: {file_path}")
                        
        logging.info(f"Found {len(training_files)} training files in {data_dir}")
        return sorted(training_files)
        
    def initialize_data_logger(self):
        """Initialize data logging system"""
        data_config = self.config.get('data', {})
        timing_config = self.config.get('timing', {})
        
        self.data_logger = DataLogger(
            output_directory=data_config.get('output_directory', 'data'),
            file_format=data_config.get('file_format', 'json'),
            drawing_delay_offset=timing_config.get('drawing_delay_offset', 0.2)
        )
        
        logging.info("Data logger initialized")
        
    def initialize_ml_components(self):
        """Initialize machine learning components"""
        self.model_trainer = ModelTrainer()
        logging.info("ML components initialized")
        
    def run_data_generation_mode(self):
        """Run the data generation interface"""
        if not PYQT_AVAILABLE:
            logging.error("PyQt5 not available. Cannot run GUI mode.")
            return
            
        try:
            app, window = create_painting_app(self.rng_device, self.eeg_bridge)
            
            # Set up data logger integration
            if self.data_logger:
                window.data_logger = self.data_logger
                
            window.show()
            
            logging.info("Data generation interface started")
            return app.exec_()
            
        except Exception as e:
            logging.error(f"Error running data generation mode: {e}")
            return 1
            
    def run_inference_mode(self, enable_streaming=False, stream_port=8765, stream_id="inference_1"):
        """Run the inference/testing interface"""
        if not PYQT_AVAILABLE:
            logging.error("PyQt5 not available. Cannot run GUI mode.")
            return
            
        try:
            app, window = create_inference_app(
                self.rng_device, 
                self.eeg_bridge,
                enable_streaming=enable_streaming,
                stream_port=stream_port,
                stream_id=stream_id
            )
            window.show()
            
            logging.info("Inference interface started")
            return app.exec_()
            
        except Exception as e:
            logging.error(f"Error running inference mode: {e}")
            return 1
            
    def run_training_mode(self, data_files: list, use_pytorch_gpu: bool = False, 
                         force_tensorflow: bool = False, force_pytorch: bool = False,
                         multi_model: bool = False, custom_variants: list = None):
        """Run ML training on collected data with flexible multi-model support"""
        
        # Auto-detect GPU and prefer PyTorch if available
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available and not force_tensorflow:
                print(f"🎮 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   Automatically enabling PyTorch GPU training")
                use_pytorch_gpu = True
                force_pytorch = True
        except ImportError:
            pass
        
        # Check if multi-model training is requested OR if PyTorch GPU is available (new default)
        if multi_model or custom_variants or (use_pytorch_gpu and not force_tensorflow):
            try:
                # Try both relative and absolute imports for compatibility
                try:
                    from .ml.multi_model_trainer import MultiModelTrainer
                except ImportError:
                    from ml.multi_model_trainer import MultiModelTrainer
                
                print("\n🚀 Starting Multi-Model Consciousness Training")
                trainer = MultiModelTrainer()
                
                if custom_variants:
                    # Train specific variants
                    print(f"🎯 Training {len(custom_variants)} specific variants")
                    trained_models = trainer.train_multiple_variants(custom_variants, data_files)
                else:
                    # Train all default variants
                    print("🌟 Training all default variants")
                    trained_models = trainer.train_default_variants(data_files)
                
                print(f"\n✅ Multi-model training completed: {len(trained_models)} models trained")
                return True
                
            except Exception as e:
                logging.error(f"Error in multi-model training: {e}")
                print(f"⚠️  Falling back to legacy training mode...")
                force_tensorflow = True
        
        # Check if hybrid training is requested (legacy mode)
        if (use_pytorch_gpu or force_pytorch or not force_tensorflow) and not force_tensorflow:
            try:
                # Try to use hybrid PyTorch training
                from ml.hybrid_training import train_consciousness_hybrid
                
                # Create training configuration
                ml_config = self.config.get('ml', {})
                arch_config = ml_config.get('architecture', {})
                
                training_config = TrainingConfig(
                    model_type=ml_config.get('model_type', 'lstm'),
                    sequence_length=ml_config.get('sequence_length', 50),
                    batch_size=ml_config.get('batch_size', 32),
                    epochs=ml_config.get('epochs', 50),
                    learning_rate=ml_config.get('learning_rate', 0.001),
                    validation_split=ml_config.get('validation_split', 0.2),
                    hidden_size=arch_config.get('hidden_size', 64),
                    num_layers=arch_config.get('num_layers', 2),
                    dropout=arch_config.get('dropout', 0.3),
                    early_stopping_patience=ml_config.get('early_stopping_patience', 8),
                    min_epochs=ml_config.get('min_epochs', 5),
                    verbose_training=ml_config.get('verbose_training', True)
                )
                
                print("🧠 " + "="*80)
                print("  HYBRID PYTORCH-TENSORFLOW CONSCIOUSNESS TRAINING")
                print("="*80)
                print(f"📊 Data files: {len(data_files)}")
                print(f"🎮 PyTorch GPU: {'Requested' if use_pytorch_gpu or force_pytorch else 'Auto-detect'}")
                print(f"🔧 Force TensorFlow: {force_tensorflow}")
                print(f"📦 Batch size: {training_config.batch_size}")
                print(f"🎯 Max epochs: {training_config.epochs}")
                print("="*80)
                print()
                
                # Train with hybrid system
                results = train_consciousness_hybrid(
                    data_files, 
                    training_config,
                    force_pytorch=force_pytorch,
                    force_tensorflow=force_tensorflow
                )
                
                print("\n" + "="*80)
                print("🎯 TRAINING RESULTS")
                print("="*80)
                print(f"Framework used: {results['framework'].upper()}")
                print(f"Device: {results['device']}")
                print(f"Final loss: {results['history']['loss'][-1]:.6f}")
                print(f"Final validation loss: {results['history']['val_loss'][-1]:.6f}")
                
                if results['framework'] == 'pytorch':
                    print(f"Model saved: {results['model_path']}")
                
                print("="*80)
                
                return 0
                
            except ImportError as e:
                print(f"⚠️  PyTorch not available: {e}")
                print("   Falling back to TensorFlow training...")
                force_tensorflow = True
            except Exception as e:
                print(f"⚠️  Hybrid training failed: {e}")
                print("   Falling back to TensorFlow training...")
                force_tensorflow = True
        
        # Fallback to original TensorFlow training
        if not self.model_trainer:
            self.initialize_ml_components()
            
        try:
            # Create training configuration
            ml_config = self.config.get('ml', {})
            arch_config = ml_config.get('architecture', {})

            training_config = TrainingConfig(
                model_type=ml_config.get('model_type', 'lstm'),
                sequence_length=ml_config.get('sequence_length', 50),
                batch_size=ml_config.get('batch_size', 8),
                epochs=ml_config.get('epochs', 50),
                learning_rate=ml_config.get('learning_rate', 0.001),
                validation_split=ml_config.get('validation_split', 0.2),
                hidden_size=arch_config.get('hidden_size', 64),
                num_layers=arch_config.get('num_layers', 2),
                dropout=arch_config.get('dropout', 0.3),
                early_stopping_patience=ml_config.get('early_stopping_patience', 8),
                min_epochs=ml_config.get('min_epochs', 5),
                verbose_training=ml_config.get('verbose_training', True)
            )

            print("🧠 " + "="*60)
            print("  CONSCIOUSNESS MODEL TRAINING (TENSORFLOW)")
            print("="*60)
            print(f"📊 Data files: {len(data_files)}")
            print(f"🔧 Model type: {training_config.model_type}")
            print(f"📐 Sequence length: {training_config.sequence_length}")
            print(f"📦 Batch size: {training_config.batch_size}")
            print(f"🎯 Max epochs: {training_config.epochs}")
            print(f"🧠 Hidden size: {training_config.hidden_size}")
            print("="*60)
            print()

            # Train models
            logging.info(f"Starting training on {len(data_files)} data files")
            results = self.model_trainer.train_models(data_files, training_config)

            logging.info("Training completed successfully")
            for mode, result in results.items():
                logging.info(f"{mode}: Final loss = {result['final_loss']:.4f}")

            return 0

        except Exception as e:
            logging.error(f"Error in training mode: {e}")
            return 1
            
    def shutdown(self):
        """Shutdown the application and cleanup resources"""
        logging.info("Shutting down application...")
        
        # Disconnect hardware
        if self.rng_device:
            self.rng_device.disconnect()
            
        if self.eeg_bridge:
            import asyncio
            asyncio.run(self.eeg_bridge.disconnect())
            
        # Stop data logging if active
        if self.data_logger and self.data_logger.session_active:
            self.data_logger.stop_session()
            
        logging.info("Application shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Consciousness Data Generation & ML App')
    
    parser.add_argument('--mode', choices=['generate', 'train', 'inference'], 
                       default='generate',
                       help='Application mode (default: generate)')
    
    parser.add_argument('--config', default='config/app_config.yaml',
                       help='Configuration file path')
    
    parser.add_argument('--data-files', nargs='+',
                       help='Data files for training mode')
    
    parser.add_argument('--data-dir', 
                       help='Directory containing training session files (searches recursively)')
    
    parser.add_argument('--no-hardware', action='store_true',
                       help='Run without hardware devices (mock mode)')
    
    parser.add_argument('--no-eeg', action='store_true',
                       help='Run without EEG device (RNG and drawing only)')
    
    parser.add_argument('--test-rng', action='store_true',
                       help='Use simulated RNG instead of TrueRNG device')
    
    parser.add_argument('--test-eeg-mode', choices=['stable', 'unstable', 'disconnected', 'connecting', 'slow_connect', 'poor_signal'],
                       default='stable', help='Mock EEG simulation mode for testing')
    
    parser.add_argument('--eeg-source', choices=['auto', 'cortex', 'mock', 'simulated'],
                       help='Force specific EEG source (overrides config)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    # Network streaming options for inference mode
    parser.add_argument('--enable-streaming', action='store_true',
                       help='Enable network streaming for inference mode')
    
    parser.add_argument('--stream-port', type=int, default=8765,
                       help='Port for inference stream server (default: 8765)')
    
    parser.add_argument('--stream-id', default='inference_1',
                       help='Stream ID for inference server (default: inference_1)')
    
    # GPU and framework options
    parser.add_argument('--use-pytorch-gpu', action='store_true',
                       help='Use PyTorch with GPU acceleration for training')
    
    parser.add_argument('--force-tensorflow', action='store_true',
                       help='Force use of TensorFlow even if PyTorch GPU is available')
    
    parser.add_argument('--force-pytorch', action='store_true',
                       help='Force use of PyTorch even if GPU not available')
    
    # Multi-model training options
    parser.add_argument('--multi-model', action='store_true',
                       help='Train multiple model variants instead of single hybrid model')
    
    parser.add_argument('--list-variants', action='store_true',
                       help='List available model variants and exit')
    
    parser.add_argument('--train-variants', nargs='+',
                       help='Train specific model variants by name')
    
    # Model management options
    parser.add_argument('--list-models', action='store_true',
                       help='List all trained models and exit')
    
    parser.add_argument('--load-model', type=str,
                       help='Load specific model for inference mode')
    
    parser.add_argument('--compare-models', action='store_true',
                       help='Enable model comparison mode in inference')
    
    args = parser.parse_args()
    
    # Handle list-variants command
    if args.list_variants:
        try:
            from .ml.model_manager import ModelManager
        except ImportError:
            from ml.model_manager import ModelManager
        manager = ModelManager()
        variants = manager.get_default_variants()
        
        print("\n🧠 Available Model Variants:")
        print("=" * 80)
        for variant in variants:
            print(f"📋 {variant.name}")
            print(f"   Framework: {variant.framework}")
            print(f"   Architecture: {variant.architecture}")
            print(f"   Features: {', '.join(variant.input_features)}")
            print(f"   Description: {variant.description}")
            print()
        
        # Also show trained models
        manager.print_model_summary()
        return 0
    
    # Handle list-models command
    if args.list_models:
        try:
            from .ml.model_manager import ModelManager
        except ImportError:
            from ml.model_manager import ModelManager
        manager = ModelManager()
        models = manager.get_available_models()
        
        if not models:
            print("\n❌ No trained models found. Train some models first!")
            print("   Use: python -m src.main --mode train --data-dir data --multi-model")
            return 0
        
        print("\n🧠 Trained Models Registry:")
        print("=" * 80)
        for name, metadata in models.items():
            config = metadata.variant_config
            print(f"📋 {name}")
            print(f"   Framework: {config.framework} ({metadata.framework_version})")
            print(f"   Architecture: {config.architecture}")
            print(f"   Features: {', '.join(config.input_features)}")
            print(f"   Hidden Size: {config.hidden_size}, Layers: {config.num_layers}")
            print(f"   Final Loss: {metadata.final_loss:.6f}, Val Loss: {metadata.final_val_loss:.6f}")
            print(f"   Epochs: {metadata.total_epochs}, GPU Used: {metadata.gpu_used}")
            print(f"   Trained: {metadata.training_time}")
            print()
        
        manager.print_model_summary()
        return 0
    
    # Create application
    app = ConsciousnessApp(args.config)
    
    # Override debug setting
    if args.debug:
        app.config['app']['debug'] = True
        app.setup_logging()
    
    try:
        # Initialize components (skip hardware for training mode)
        if args.mode != 'train':
            if not args.no_hardware and not args.test_rng:
                # Override EEG source if specified
                if args.eeg_source:
                    app.config['eeg'] = app.config.get('eeg', {})
                    app.config['eeg']['source'] = args.eeg_source
                    
                # Override mock EEG mode for testing
                if args.test_eeg_mode:
                    app.config['eeg'] = app.config.get('eeg', {})
                    app.config['eeg']['mock'] = app.config['eeg'].get('mock', {})
                    app.config['eeg']['mock']['mode'] = args.test_eeg_mode
                    
                app.initialize_hardware()
            else:
                if args.test_rng or args.no_hardware:
                    logging.info("Running with simulated RNG device")
                    # Create mock RNG with config rate or specified rate
                    from hardware.truerng_v3 import MockTrueRNG
                    config_rate = app.config.get('hardware', {}).get('truerng', {}).get('generation_rate_kbps', 312.0)
                    rng_rate = getattr(args, 'rng_rate', config_rate)
                    app.rng_device = MockTrueRNG(generation_rate_kbps=rng_rate)
                    app.rng_device.connect()
                
                if args.no_eeg or args.no_hardware:
                    logging.info("Running without EEG device")
                    app.eeg_bridge = None
                else:
                    # Create EEG bridge with mock configuration
                    eeg_config = load_eeg_config()
                    eeg_config['source'] = 'mock'
                    eeg_config['mock']['mode'] = getattr(args, 'test_eeg_mode', 'stable')
                    
                    app.eeg_bridge = EEGBridge(eeg_config)
                    import asyncio
                    connected = asyncio.run(app.eeg_bridge.connect())
                    
                    if connected:
                        connection_info = app.eeg_bridge.get_connection_info()
                        logging.info(f"Mock EEG connected: {connection_info.source.value} ({connection_info.device_info.get('mode', 'unknown')})")
                    else:
                        logging.warning("Mock EEG connection failed")
                        app.eeg_bridge = None
                
            # Override EEG bridge if --no-eeg flag is used
            if args.no_eeg:
                if app.eeg_bridge:
                    import asyncio
                    asyncio.run(app.eeg_bridge.disconnect())
                app.eeg_bridge = None
                logging.info("EEG device disabled by --no-eeg flag")
            
            # Initialize data logger (requires hardware)
            app.initialize_data_logger()
        else:
            # Training mode - no hardware needed
            logging.info("Training mode: skipping hardware initialization (loading from saved session files)")
            app.rng_device = None
            app.eeg_bridge = None
        
        # Always initialize ML components (they don't require hardware)
        app.initialize_ml_components()
        
        # Handle data directory for training
        data_files = args.data_files
        if args.mode == 'train':
            if args.data_dir:
                data_files = app.find_training_files(args.data_dir)
                logging.info(f"Found {len(data_files)} training files in {args.data_dir}")
            elif not args.data_files:
                logging.error("Training mode requires either --data-files or --data-dir argument")
                return 1
        
        # Run selected mode
        if args.mode == 'generate':
            return app.run_data_generation_mode()
            
        elif args.mode == 'inference':
            return app.run_inference_mode(
                enable_streaming=args.enable_streaming,
                stream_port=args.stream_port,
                stream_id=args.stream_id
            )
            
        elif args.mode == 'train':
            # Prepare training options
            custom_variants = None
            
            if args.train_variants:
                # Create custom variants based on specified names
                try:
                    from .ml.model_manager import ModelManager
                except ImportError:
                    from ml.model_manager import ModelManager
                manager = ModelManager()
                all_variants = manager.get_default_variants()
                variant_dict = {v.name: v for v in all_variants}
                
                custom_variants = []
                for name in args.train_variants:
                    if name in variant_dict:
                        custom_variants.append(variant_dict[name])
                    else:
                        print(f"⚠️  Unknown variant: {name}")
                
                if not custom_variants:
                    print("❌ No valid variants specified")
                    return 1
            
            return app.run_training_mode(
                data_files,
                use_pytorch_gpu=args.use_pytorch_gpu,
                force_tensorflow=args.force_tensorflow,
                force_pytorch=args.force_pytorch,
                multi_model=args.multi_model,
                custom_variants=custom_variants
            )
            
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
        return 0
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1
        
    finally:
        app.shutdown()


if __name__ == "__main__":
    sys.exit(main())