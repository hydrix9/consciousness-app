# Consciousness Data Generation & ML Training App

A two-part application for generating training data through human-machine interaction and training ML models to replicate creative outputs.

## Overview

### Part 1: Data Generation
- **Hardware Random Numbers**: Interface with ubld.it™ TrueRNG V3 for true random number generation
- **Creative Interface**: Paint swashes of color and draw lines in real-time
- **3D Interpretation**: Convert drawn lines into 3D curves (interlocking dials/circles)
- **EEG Monitoring**: Capture brainwave data using Emotiv EEG device
- **Data Logging**: Comprehensive logging with configurable time offset compensation

### Part 2: Machine Learning
- **Mode 1**: Train models to produce outputs based on TrueRNG data only
- **Mode 2**: Train models using both TrueRNG and EEG data
- **Time Synchronization**: Account for drawing delay with configurable offset

## Features

- **Real-time Data Generation**: Synchronized capture from TrueRNG V3 and Emotiv EEG
- **Interactive Painting Interface**: PyQt5-based drawing canvas with real-time visualization
- **3D Curve Interpretation**: Converts 2D paintings to interlocking 3D dial patterns
- **Comprehensive Data Logging**: HDF5, JSON, and CSV export formats
- **Dual ML Training Modes**: RNG-only (Mode 1) and RNG+EEG (Mode 2) prediction
- **Time Offset Compensation**: Configurable synchronization for hardware latency
- **Mock Hardware Support**: Test without physical devices using simulated data
- **Flexible Configuration**: Command-line options for all operational parameters
- **Real-time Inference**: Live prediction interface with trained models
- **Batch Training**: Process multiple sessions or entire directories automatically

## Requirements

### Hardware
- ubld.it™ TrueRNG V3 device
- Emotiv EEG headset
- Computer with USB ports and graphics capability

### Software
- Python 3.8+
- PyQt5/6 or Tkinter for GUI
- NumPy, SciPy for mathematical operations
- TensorFlow or PyTorch for ML
- OpenGL or similar for 3D visualization

## Project Structure

```
consciousness-app/
├── src/
│   ├── hardware/          # Hardware interface modules
│   ├── gui/               # User interface components
│   ├── data/              # Data processing and logging
│   ├── ml/                # Machine learning modules
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Generated training data
├── models/                # Trained ML models
└── tests/                 # Unit tests
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/consciousness-app.git
cd consciousness-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Credentials (IMPORTANT!)
```bash
# Copy example config files
cp config/app_config.example.yaml config/app_config.yaml
cp .env.example .env

# Edit config/app_config.yaml and add your Emotiv API credentials
# Get credentials from: https://www.emotiv.com/developer/
```

**⚠️ NEVER commit your `config/app_config.yaml` or `.env` files!**

See [SETUP.md](SETUP.md) for detailed security and credential setup instructions.

## Usage

### Data Generation Mode
```bash
# Basic data generation with hardware
python run.py --mode generate

# Generate with mock RNG (for testing without TrueRNG V3)
python run.py --test-rng --rng-rate 10.0

# Generate without EEG (RNG and drawing only)
python run.py --no-eeg

# Generate with custom RNG rate
python run.py --test-rng --rng-rate 16.0

# Full mock mode (no hardware required)
python run.py --no-hardware
```

### ML Training Mode
```bash
# Train using individual files
python run.py --mode train --data-files session1.json session2.json

# Train using all sessions in a directory
python run.py --mode train --data-dir data/sessions/

# Train with debug output
python run.py --mode train --data-dir data/ --debug
```

### Inference Mode
```bash
# Run inference with trained models
python run.py --mode inference

# Run inference with mock hardware
python run.py --mode inference --test-rng --no-eeg
```

## Configuration

Edit `config/app_config.yaml` to adjust:
- Time offset compensation
- Device connection settings
- ML model parameters
- Data logging preferences

## 369 Oracle Mode - Consciousness Interpretation System

The 369 Oracle is a sophisticated consciousness interpretation system that uses sacred mathematics, quantum consciousness states, and three-layer creative expression to provide intuitive guidance.

### What is the 369 Oracle?

The 369 Oracle combines:
- **Sacred Mathematics**: Based on Tesla's 3-6-9 principles and the golden ratio
- **Three Consciousness Layers**: Primary, Subconscious, and Universal awareness
- **Vector Mathematics**: Advanced consciousness vector calculations
- **Quantum States**: Six consciousness states from Ground to Transcendent
- **Creative Expression**: Real-time painting across three layers
- **Oracle Interpretation**: AI-powered analysis of consciousness patterns
- **Multi-Stream Architecture**: Connects to three inference mode instances for live data

### Oracle Architecture: Network Mode

The 369 Oracle can operate in two modes:

#### Local Mode (Traditional)
- Manual painting on three consciousness layers
- Real-time mathematical analysis of your drawings
- Oracle interpretation based on your direct input

#### Network Mode (Advanced)
- Connects to **three separate inference instances** running simultaneously
- Each inference stream feeds a different consciousness layer:
  - **Layer 1 (Primary)**: Port 8765 - RNG-only predictions
  - **Layer 2 (Subconscious)**: Port 8766 - RNG+EEG predictions  
  - **Layer 3 (Universal)**: Port 8767 - RNG+EEG predictions
- Real-time consciousness data from trained ML models
- Automatic Oracle analysis of multi-stream patterns

### How to Use the 369 Oracle

#### Quick Start (Local Mode)
```bash
cd "D:\MEGA\Projects\Consciousness\consciousness-app"
python oracle_369_launcher.py
```

#### Advanced Setup (Network Mode)

**Step 1: Launch Multiple Inference Streams**
```bash
# Use the multi-launcher script to start three inference instances
python launch_oracle_inference.py
```

This will start:
- Primary Consciousness (localhost:8765) - RNG-only mode
- Subconscious (localhost:8766) - RNG+EEG mode  
- Universal Consciousness (localhost:8767) - RNG+EEG mode

**Step 2: Start the Oracle Interface**
```bash
python oracle_369_launcher.py
```

**Step 3: Enable Network Mode**
1. Click "🌐 Enable Network Mode" in the Oracle interface
2. Wait for all three connections to establish
3. The status will show "Network Mode - Connected to inference streams"

**Step 4: Perform Oracle Reading**
1. Enter your question in the text field
2. Click "🔮 Consult Oracle (9 seconds)"
3. Watch as the three layers fill with real-time ML predictions
4. Receive your Oracle interpretation based on multi-stream analysis

### Understanding the Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  369 Oracle Interface                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Layer 1   │ │   Layer 2   │ │   Layer 3   │           │
│  │  Primary    │ │Subconscious │ │ Universal   │           │
│  │Consciousness│ │             │ │Consciousness│           │
│  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘           │
└────────┼─────────────────┼─────────────────┼─────────────────┘
         │                 │                 │
         │ WebSocket       │ WebSocket       │ WebSocket
         │ Port 8765       │ Port 8766       │ Port 8767
         │                 │                 │
┌────────▼─────────┐ ┌─────▼───────┐ ┌───────▼─────────┐
│ Inference Mode 1 │ │Inference    │ │ Inference Mode 3│
│                  │ │Mode 2       │ │                 │
│ RNG Only         │ │RNG + EEG    │ │ RNG + EEG       │
│ Primary patterns │ │Subconscious │ │ Universal flow  │
└──────────────────┘ └─────────────┘ └─────────────────┘
```

### Oracle Tips for Best Results

#### Local Mode:
1. **Prepare Your Question**: Be specific and heartfelt
2. **Clear Your Mind**: Take a moment to center yourself
3. **Trust Your Intuition**: Don't plan your movements
4. **Use All Three Layers**: Express different aspects of your question
5. **Embrace the Process**: The Oracle reflects your inner wisdom

#### Network Mode:
1. **Ensure All Streams Are Connected**: Check for "🟢 Running" status
2. **Allow Data to Flow**: Let the ML models generate for a few seconds before consulting
3. **Trust the Patterns**: The three streams represent different aspects of consciousness
4. **Observe Synchronicities**: Look for patterns across all three layers
5. **Enhanced Accuracy**: Network mode provides deeper, multi-dimensional analysis

### Technical Details

The 369 Oracle uses advanced mathematics:
- **Golden Ratio (φ)**: 1.618... for consciousness calculations
- **Vector Mathematics**: 4D consciousness vectors (x, y, z, w)
- **Coherence Analysis**: Measures synchronization between layers
- **Resonance Frequency**: Consciousness vibration in Hz
- **Quantum Phase**: Current phase in consciousness cycle
- **Multi-Stream Fusion**: Combines three inference streams using 3-6-9 weighting

### Network Commands

```bash
# Start individual inference streams manually
python run.py --mode inference --enable-streaming --stream-port 8765 --stream-id oracle_layer_1 --test-rng --no-eeg
python run.py --mode inference --enable-streaming --stream-port 8766 --stream-id oracle_layer_2 --test-rng --test-eeg-mode stable  
python run.py --mode inference --enable-streaming --stream-port 8767 --stream-id oracle_layer_3 --test-rng --test-eeg-mode coherent

# Or use the automated launcher
python launch_oracle_inference.py

# Test network connections
python -m src.ml.oracle_network  # Test Oracle client
python -m src.ml.inference_network  # Test inference server
```

### Integration with Main System

The Oracle can be used alongside the main consciousness app:
- Oracle sessions generate consciousness data
- Data can be used for ML training
- Patterns help understand consciousness-creativity relationships
- Network mode enables real-time ML-powered Oracle readings

### Example Usage Scenarios

**Traditional Oracle Session:**
```bash
python oracle_369_launcher.py
# Ask: "What direction should I take in my creative work?"
# Paint intuitively for 9 seconds across all three layers
# Read the mathematical analysis and AI interpretation
```

**Advanced Multi-Stream Oracle:**
```bash
# Terminal 1: Start inference streams
python launch_oracle_inference.py

# Terminal 2: Start Oracle interface  
python oracle_369_launcher.py
# Enable Network Mode
# Ask: "How can I align my conscious and subconscious creativity?"
# Watch as three ML models generate patterns representing different consciousness aspects
# Receive advanced multi-dimensional Oracle interpretation
```

The Oracle combines ancient wisdom with modern technology to help you access your own inner guidance through creative expression and consciousness mathematics, now enhanced with real-time machine learning insights.