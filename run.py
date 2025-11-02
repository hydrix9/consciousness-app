#!/usr/bin/env python3
"""
Consciousness Data Generation & ML Training App
===============================================

Quick Start Guide:

1. Install Dependencies:
   pip install -r requirements.txt

2. Run Data Generation Mode (default):
   python run.py

3. Run Inference Mode:
   python run.py --mode inference

4. Train Models (using individual files):
   python run.py --mode train --data-files data/session1.json data/session2.json

5. Train Models (using directory):
   python run.py --mode train --data-dir data/

6. Run without hardware (mock mode):
   python run.py --no-hardware

7. Test RNG generation without TrueRNG device:
   python run.py --test-rng --rng-rate 12.0

8. Run without EEG (RNG and drawing only):
   python run.py --no-eeg

Command Line Options:
  --mode {generate,train,inference}  Application mode
  --config CONFIG                    Configuration file path
  --data-files FILES                 Individual data files for training
  --data-dir DIRECTORY               Directory containing training sessions
  --no-hardware                      Run with all mock devices
  --no-eeg                          Run without EEG device
  --test-rng                        Use simulated RNG (for testing)
  --rng-rate RATE                   RNG rate in kilobits/second (default: 8.0)
  --debug                           Enable debug logging

Hardware Setup:
- Connect TrueRNG V3 device via USB
- Set up Emotiv EEG headset  
- Configure credentials in config/app_config.yaml

Examples:
  # Generate data with mock RNG at 16 kbps, no EEG
  python run.py --test-rng --rng-rate 16.0 --no-eeg
  
  # Train on all sessions in data folder
  python run.py --mode train --data-dir data/
  
  # Run inference with real hardware
  python run.py --mode inference

For detailed instructions, see README.md
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    sys.exit(main())