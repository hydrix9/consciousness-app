# Consciousness App - Initial Commit

## 🎨 Core Features

### Data Generation
- Real-time painting interface with PyQt5
- Hardware integration (TrueRNG V3, Emotiv EEG)
- Mock hardware support for testing
- 3D curve interpretation (interlocking dials)
- Multi-layer consciousness painting system
- Pocket dimension navigation
- HDF5/JSON data logging

### Machine Learning
- PyTorch GPU-accelerated training
- Dual training modes (RNG-only, RNG+EEG)
- Real-time inference with trained models
- Model registry and versioning
- Advanced data augmentation
- HDF5-based data loading pipeline

### 369 Oracle System
- Sacred mathematics integration
- Three-layer consciousness interface
- Vector-based consciousness calculations
- Network mode with multi-stream inference
- WebSocket-based real-time data streaming
- AI-powered Oracle interpretations

## 🎨 Enhanced Painting Features

### Color Palette
- 8 mystical colors with keyboard shortcuts (1-8)
- White color for technical visualization
- Visual feedback on color selection
- Optimized for workflow

### Visualization
- Interlocking dial visualization overlay
- White stroke rendering for 3D geometry
- Transparent overlay compositing
- Real-time curve-to-dial conversion

### Multi-Dimensional Navigation
- 3 consciousness layers
- Unlimited pocket dimensions per layer
- Visual dimension indicators
- Smooth layer/dimension transitions

## 🔧 Technical Stack

- Python 3.8+
- PyTorch (GPU accelerated)
- PyQt5 GUI framework
- HDF5 for data storage
- WebSocket for networking
- YAML configuration

## 📁 Project Structure

```
consciousness-app/
├── src/
│   ├── hardware/      # TrueRNG V3, Emotiv EEG interfaces
│   ├── gui/           # Painting interface, visualizations
│   ├── data/          # Data logging and HDF5 handling
│   ├── ml/            # PyTorch models, training, inference
│   └── utils/         # 3D curves, mathematics
├── config/            # Configuration templates
├── models/            # Trained models (not committed)
├── data/              # Session data (not committed)
└── tests/             # Test scripts and validation

## 🔐 Security

- Credentials managed via .env and config files
- Sensitive files excluded from git
- Example config files provided
- See SETUP.md for security guidelines

## 📝 Documentation

- README.md - Main documentation
- SETUP.md - Security and credential setup
- Multiple feature documentation files
- Code comments and docstrings

## 🎯 Next Steps

1. Copy example config files
2. Add your Emotiv credentials
3. Run in test mode: `python run.py --test-rng --no-eeg --debug`
4. Generate training data
5. Train models
6. Explore the 369 Oracle

## ⚠️ Important Notes

- This is the initial public release
- Sensitive credentials removed
- Large model files excluded
- Personal session data excluded
- See .gitignore for complete exclusion list
